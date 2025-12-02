import os
import sys
import numpy as np
import cv2
import json
import pandas as pd
import lungs_finder as lf
from pathlib import Path
import pydicom
from dicompylercore import dicomparser
from tensorflow.keras.models import load_model
import argparse
from datetime import datetime, timezone, timedelta

def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate cardiac pathologies from DICOM or other image formats.")
    parser.add_argument("input_path", help="Path to the input image file or directory.")
    parser.add_argument("model_path", help="Path to the trained Keras model (.h5).")
    parser.add_argument("config_path", help="Path to the JSON configuration file containing model specifications.")
    parser.add_argument("-o", "--output_path", help="Path to the output directory for JSON results. Defaults to the current directory.", default=".")
    parser.add_argument("--use_labels", action="store_true", help="Display labels on the output images. Defaults to False.")
    args = parser.parse_args()
    return args.input_path, args.model_path, args.config_path, args.use_labels, args.output_path


def proportional_resize(image, max_side):
    """
    Resizes an image proportionally to fit within a specified maximum side length.

    Args:
        image: The input image to be resized.
        max_side: The maximum side length of the resized image.

    Returns:
        The resized image.
    """    
    if image.shape[0] > max_side or image.shape[1] > max_side:
        if image.shape[0] > image.shape[1]:
            height = max_side
            width = int(height / image.shape[0] * image.shape[1])
        else:
            width = max_side
            height = int(width / image.shape[1] * image.shape[0])
    else:
        height = image.shape[0]
        width = image.shape[1]

    return cv2.resize(image, (width, height))


def scan():
    input_path, model_path, config_path, use_labels, output_path = parse_arguments()
    
    output_path_obj = Path(output_path)
    if not os.path.exists(output_path_obj):
        os.makedirs(output_path_obj)
        
    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load variables from the configuration JSON file
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            json_in = json.load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    img_width = json_in.get('img_width')
    img_height = json_in.get('img_height')
    inverted_labels = json_in.get('Classification')
    lb = json_in.get('Labels')

    if not all([img_width, img_height, inverted_labels, lb]):
        print("Error: Configuration file is missing required fields (img_width, img_height, Classification, Labels).")
        return

    print('img_width:', img_width)
    print('img_height:', img_height)
    print('Classification:', inverted_labels)
    print('Labels:', lb)
    
    files_to_process = []
    if os.path.isdir(input_path):
        for path, directories, files in os.walk(input_path):
            files_to_process.extend([os.path.join(path, file) for file in files if not file.startswith(".")])
    elif os.path.isfile(input_path):
        files_to_process.append(input_path)
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")
        return

    for full_file_path in files_to_process:
            path, file = os.path.split(full_file_path)
            filename = Path(file).stem
            print("Processing file:", file)

            _, extension = os.path.splitext(file)
            
            dicom_info = {}
            accnum = filename # Default for non-dicom

            if extension == ".dcm":
                try:
                    dataset = pydicom.dcmread(full_file_path)
                    
                    patient_name_str = str(dataset.get('PatientName', 'N/A'))
                    dicom_info['PatientName'] = patient_name_str
                    
                    if '^' in patient_name_str:
                        parts = patient_name_str.split('^')
                        dicom_info['PatientFamilyName'] = parts[0]
                        dicom_info['PatientGivenName'] = parts[1] if len(parts) > 1 else ''
                    else:
                        dicom_info['PatientFamilyName'] = patient_name_str
                        dicom_info['PatientGivenName'] = ''

                    dicom_info['AccessionNumber'] = str(dataset.get('AccessionNumber', 'N/A'))

                    patient_id = 'N/A'
                    issuer_of_patient_id = 'N/A'
                    try:
                        modified_attributes_sequence = dataset.get((0x0400, 0x0550))
                        if modified_attributes_sequence and len(modified_attributes_sequence) > 0:
                            item = modified_attributes_sequence[0]
                            if (0x0010, 0x0020) in item:
                                patient_id = str(item[0x0010, 0x0020].value)
                            if (0x0010, 0x0021) in item:
                                issuer_of_patient_id = str(item[0x0010, 0x0021].value)
                    except Exception as e:
                        print(f"Could not read Modified Attributes Sequence: {e}")

                    if patient_id == 'N/A':
                        patient_id = str(dataset.get('PatientID', 'N/A'))

                    if dicom_info['AccessionNumber'] == 'N/A':
                        dicom_info['AccessionNumber'] = patient_id

                    dicom_info['PatientID'] = patient_id
                    dicom_info['IssuerOfPatientID'] = issuer_of_patient_id
                    dicom_info['PatientBirthDate'] = str(dataset.get('PatientBirthDate', 'N/A'))
                    dicom_info['PatientSex'] = str(dataset.get('PatientSex', 'N/A'))
                    
                    dicom_info['StudyInstanceUID'] = str(dataset.get('StudyInstanceUID', 'N/A'))
                    dicom_info['SeriesInstanceUID'] = str(dataset.get('SeriesInstanceUID', 'N/A'))
                    dicom_info['SOPInstanceUID'] = str(dataset.get('SOPInstanceUID', 'N/A'))
                    dicom_info['SOPClassUID'] = str(dataset.get('SOPClassUID', 'N/A'))

                    dicom_info['StudyID'] = str(dataset.get('StudyID', 'N/A'))
                    dicom_info['StudyDescription'] = str(dataset.get('StudyDescription', 'N/A'))
                    dicom_info['SeriesDescription'] = str(dataset.get('SeriesDescription', 'N/A'))
                    dicom_info['SeriesNumber'] = str(dataset.get('SeriesNumber', 'N/A'))
                    dicom_info['InstanceNumber'] = str(dataset.get('InstanceNumber', 'N/A'))
                    dicom_info['BodyPartExamined'] = str(dataset.get('BodyPartExamined', 'N/A'))
                    
                    dicom_info['Modality'] = str(dataset.get('Modality', 'N/A'))
                    dicom_info['InstitutionName'] = str(dataset.get('InstitutionName', 'N/A'))
                    dicom_info['StationName'] = str(dataset.get('StationName', 'N/A'))
                    dicom_info['StudyDate'] = str(dataset.get('StudyDate', 'N/A'))
                    dicom_info['StudyTime'] = str(dataset.get('StudyTime', 'N/A'))
                    dicom_info['SeriesDate'] = str(dataset.get('SeriesDate', 'N/A'))
                    dicom_info['SeriesTime'] = str(dataset.get('SeriesTime', 'N/A'))
                    acquisition_datetime = 'N/A'
                    try:
                        acq_date = dataset.get('AcquisitionDate', '')
                        acq_time = dataset.get('AcquisitionTime', '')
                        if acq_date and acq_time:
                            dt_str = f"{acq_date}{acq_time}"
                            # Handle fractional seconds
                            if '.' in dt_str:
                                dt_obj = datetime.strptime(dt_str, '%Y%m%d%H%M%S.%f')
                            else:
                                dt_obj = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
                            acquisition_datetime = dt_obj.isoformat()
                    except Exception as e:
                        print(f"Could not format AcquisitionDateTime: {e}")
                    dicom_info['AcquisitionDateTime'] = acquisition_datetime

                    dicom_info['Manufacturer'] = str(dataset.get('Manufacturer', 'N/A'))
                    dicom_info['ManufacturerModelName'] = str(dataset.get('ManufacturerModelName', 'N/A'))
                    dicom_info['DeviceSerialNumber'] = str(dataset.get('DeviceSerialNumber', 'N/A'))
                    dicom_info['SoftwareVersions'] = str(dataset.get('SoftwareVersions', 'N/A'))
                    
                    try:
                        dicom_info['ImplementationVersionName'] = str(dataset.file_meta.get('ImplementationVersionName', 'N/A'))
                    except AttributeError:
                        dicom_info['ImplementationVersionName'] = 'N/A'

                    dicom_info['SpatialResolution'] = str(dataset.get('PixelSpacing', 'N/A'))
                    dicom_info['DetectorManufacturerModelName'] = str(dataset.get('DetectorManufacturerModelName', 'N/A'))

                    image_params = {
                        'Rows': str(dataset.get('Rows', 'N/A')),
                        'Columns': str(dataset.get('Columns', 'N/A')),
                        'PixelSpacing': str(dataset.get('PixelSpacing', 'N/A')),
                        'WindowCenter': str(dataset.get('WindowCenter', 'N/A')),
                        'WindowWidth': str(dataset.get('WindowWidth', 'N/A')),
                        'PhotometricInterpretation': str(dataset.get('PhotometricInterpretation', 'N/A'))
                    }
                    dicom_info['ImageParameters'] = image_params
                    
                    accnum = dicom_info['AccessionNumber'] if dicom_info['AccessionNumber'] != 'N/A' else filename

                    parsed = dicomparser.DicomParser(full_file_path)
                    image = np.array(parsed.GetImage(), dtype=np.uint8)

                    if parsed.GetImageData().get("photometricinterpretation") == "MONOCHROME1":
                        image = 255 - image

                    image = cv2.equalizeHist(image)
                    image = cv2.medianBlur(image, 3)
                except Exception as e:
                    print(f"Error reading DICOM file {file}: {e}")
                    continue

            elif extension.lower() in [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", ".jpe", ".png",
                               ".tiff", ".tif"]:
                image = cv2.imread(full_file_path, 0)
                if image is None:
                    print(f"Error reading image file {file}")
                    continue
                image = cv2.equalizeHist(image)
                image = cv2.medianBlur(image, 3)
            else:
                continue

            scaled_image = proportional_resize(image, 640)
            
            # Lung detection using lungs_finder module
            right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
            left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
            right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
            left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
            right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
            left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
            color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

            # Draw rectangles for detected lungs
            if right_lung_hog_rectangle is not None:
                x, y, width, height = right_lung_hog_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
                if use_labels:
                    cv2.putText(color_image, "HOG Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (59, 254, 211), 1)

            if left_lung_hog_rectangle is not None:
                x, y, width, height = left_lung_hog_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
                if use_labels:
                    cv2.putText(color_image, "HOG Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (59, 254, 211), 1)

            if right_lung_lbp_rectangle is not None:
                x, y, width, height = right_lung_lbp_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
                if use_labels:
                    cv2.putText(color_image, "LBP Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (130, 199, 0), 1)

            if left_lung_lbp_rectangle is not None:
                x, y, width, height = left_lung_lbp_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
                if use_labels:
                    cv2.putText(color_image, "LBP Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (130, 199, 0), 1)

            if right_lung_haar_rectangle is not None:
                x, y, width, height = right_lung_haar_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
                if use_labels:
                    cv2.putText(color_image, "HAAR Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (245, 199, 75), 1)

            if left_lung_haar_rectangle is not None:
                x, y, width, height = left_lung_haar_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
                if use_labels:
                    cv2.putText(color_image, "HAAR Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (245, 199, 75), 1)

            found_lungs = lf.get_lungs(scaled_image)
            
            image_to_process = scaled_image # Default to using the scaled image
            
            # Check if the crop is good enough to use
            if found_lungs is not None and found_lungs.size > 0:
                h, w = found_lungs.shape[:2]
                if w > 0 and h > 0:
                    aspect_ratio = h / w
                    if aspect_ratio >= 0.25 and aspect_ratio <= 4.0:
                        image_to_process = found_lungs # Use the crop
                        print(f"Using cropped lung image (shape: {found_lungs.shape}).")
                    else:
                        print(f"Cropped image is too thin (shape: {found_lungs.shape}), using original scaled image.")
                else:
                    print(f"Cropped image has invalid dimensions (shape: {found_lungs.shape}), using original scaled image.")
            else:
                print("Lungs not found, using original scaled image.")
            
            # Create a temporary file path for the processed lung image
            temp_output_dir = output_path_obj / "temp_processed"
            temp_output_dir.mkdir(exist_ok=True)
            output_file = temp_output_dir / f"{filename}.jpg"

            if image_to_process is not None and image_to_process.size > 0:
                try:
                    if cv2.imwrite(str(output_file), image_to_process):
                        
                        input_image = cv2.imread(str(output_file))
                        input_image_resized = cv2.resize(input_image, (img_width, img_height))
                        input_image_normalized = input_image_resized / 255.0
                        input_image_reshaped = np.reshape(input_image_normalized, (1, img_width, img_height, 3))

                        input_prediction = model.predict(input_image_reshaped)
                        prediction = input_prediction.tolist()
                        
                        input_pred_label = np.argmax(input_prediction)
                        
                        predicted_label_str = 'Unknown Label'
                        if str(input_pred_label) in inverted_labels:
                            predicted_label_str = inverted_labels[str(input_pred_label)]
                            print(input_pred_label, predicted_label_str)
                        else:
                            print('Unknown Label')

                        prediction_accuracy = np.max(input_prediction).tolist()

                        # Get current time for PredictionDateTime
                        now = datetime.now(timezone(timedelta(hours=2))) # Assuming UTC+2
                        prediction_datetime = now.isoformat()

                        final_json_output = {
                            "Classification": predicted_label_str,
                            "PredictionAccuracy": prediction_accuracy,
                            "ClassificationEncoding": {
                                "Labels": lb,
                                "ClassificationMap": inverted_labels
                            },
                            "PredictionProbabilities": prediction[0],
                            "PredictionDateTime": prediction_datetime,
                        }

                        if dicom_info:
                            final_json_output["DicomInfo"] = dicom_info
                        else:
                            final_json_output["FileInfo"] = {"FileName": file}
                        
                        json_filename = accnum if accnum != filename else filename
                        
                        json_output_path = output_path_obj / f"{json_filename}.json"
                        with open(json_output_path, 'w') as f:
                            json.dump(final_json_output, f, indent=4)
                        print(f"JSON result saved to: {json_output_path}")
                        
                    else:
                        print(f"Error writing temporary image {filename}")
                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")

            if cv2.waitKey(1) == 27:
                break

if __name__ == "__main__":
    sys.exit(scan())
