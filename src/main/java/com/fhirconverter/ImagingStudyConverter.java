package com.fhirconverter;

import ca.uhn.fhir.context.FhirContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.hl7.fhir.r5.model.*;
import org.hl7.fhir.r5.model.ImagingStudy.ImagingStudySeriesComponent;
import org.hl7.fhir.r5.model.ImagingStudy.ImagingStudySeriesInstanceComponent;
import org.hl7.fhir.r5.model.ImagingStudy.ImagingStudySeriesPerformerComponent;
import org.hl7.fhir.r5.model.DiagnosticReport.DiagnosticReportStatus;
import org.hl7.fhir.r5.model.StringType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.hl7.fhir.r5.model.Provenance;
import org.hl7.fhir.r5.model.Provenance.ProvenanceAgentComponent;
import org.hl7.fhir.r5.model.Organization;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.UUID;

/**
 * Converter for transforming imaging study JSON data to FHIR ImagingStudy
 * resources.
 * Handles DICOM metadata and AI prediction results.
 */
public class ImagingStudyConverter {

    private static final Logger logger = LoggerFactory.getLogger(ImagingStudyConverter.class);
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyyMMdd");
    private static final SimpleDateFormat DATE_TIME_FORMAT = new SimpleDateFormat("yyyyMMddHHmmss");
    private static final String DICOM_ORG_ROOT = "1.2.826.0.1.3680043.8.4987"; // Example organization root
    private static final String DICOM_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2"; // CT Image Storage SOP Class UID
    private static final String DICOM_DX_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.1.1"; // Digital X-Ray Image Storage
                                                                                        // SOP Class UID

    private final FhirContext fhirContext;
    private final ObjectMapper objectMapper;

    public ImagingStudyConverter() {
        this.fhirContext = FhirContext.forR5();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Extracts AI model version/name from multiple possible field names.
     * Supports: AIModelVersion, ModelVersion, AIModel, ModelName, AIModelName, Model, AIName
     */
    private String extractAIModelVersion(JsonNode imagingData) {
        String[] possibleFields = {
            "AIModelVersion", "ModelVersion", "AIModel", 
            "ModelName", "AIModelName", "Model", "AIName",
            "AI_Model_Version", "AI_Model_Name", "model_version", "model_name"
        };
        
        for (String field : possibleFields) {
            if (imagingData.has(field)) {
                String value = imagingData.path(field).asText();
                if (!value.isEmpty() && !value.equals("N/A")) {
                    return value;
                }
            }
        }
        return null;
    }

    /**
     * Extracts processing/inference timestamp from multiple possible field names.
     * Supports: ProcessingTimestamp, InferenceTimestamp, ModelProcessingTime, ComputeTime, etc.
     * Falls back to PredictionDateTime if available.
     */
    private String extractProcessingTimestamp(JsonNode imagingData) {
        String[] possibleFields = {
            "ProcessingTimestamp", "InferenceTimestamp", "ModelProcessingTime",
            "ComputeTime", "ProcessingTime", "ExecutionTime", 
            "processing_timestamp", "inference_timestamp", "model_processing_time"
        };
        
        for (String field : possibleFields) {
            if (imagingData.has(field)) {
                String value = imagingData.path(field).asText();
                if (!value.isEmpty() && !value.equals("N/A")) {
                    return value;
                }
            }
        }
        
        // Fallback to PredictionDateTime if available
        if (imagingData.has("PredictionDateTime")) {
            String value = imagingData.path("PredictionDateTime").asText();
            if (!value.isEmpty() && !value.equals("N/A")) {
                return value;
            }
        }
        
        return null;
    }

    /**
     * Creates an Organization resource representing the AI system (model + training provenance)
     */
    /**
     * Creates a Device resource representing the AI system (model + training provenance)
     */
    private Device createAIDevice(JsonNode imagingData) {
        // Attempt to extract model and training info from multiple possible fields
        String modelVersion = extractAIModelVersion(imagingData);

        // If no useful AI metadata present, still create a minimal Device to act as responsible party
        Device device = new Device();
        device.setId(UUID.randomUUID().toString());
        
        String deviceName = null;
        if (modelVersion != null && !modelVersion.isBlank()) {
            deviceName = modelVersion;
        } else {
            // fallback: if ClassificationEncoding contains labels, use a generic name
            if (imagingData.has("ClassificationEncoding")) {
                deviceName = "AI_Analyzer";
            } else {
                // If there's not even AI data, don't create device
                return null;
            }
        }

        // Set Device Name
        device.addName()
                .setValue(deviceName)
                // MODEL_NAME is missing in this HAPI FHIR version, using REGISTEREDNAME as fallback
                .setType(org.hl7.fhir.r5.model.Enumerations.DeviceNameType.REGISTEREDNAME);

        // Set Version
        if (modelVersion != null && !modelVersion.isBlank()) {
            device.addVersion().setValue(modelVersion);
        }
        
        // Set Type to SNOMED "Software application"
        device.addType(new CodeableConcept().addCoding(new Coding("http://snomed.info/sct", "702499008", "Software application")));

        // Add identifier for model version if available
        if (modelVersion != null && !modelVersion.isBlank()) {
            device.addIdentifier()
                    .setSystem("urn:ai:model-version")
                    .setValue(modelVersion);
        }

        // Set owner - the hospital/vendor organization that owns the AI system
        device.setOwner(new Reference()
                .setDisplay("Fondazione Toscana Gabriele Monasterio"));

        return device;
    }

    /**
     * Creates a Library resource representing the Training Data
     */
    private Library createTrainingDataLibrary(JsonNode imagingData) {
        String trainingDataset = null;
        if (imagingData.has("TrainingDataVersion")) {
            trainingDataset = imagingData.path("TrainingDataVersion").asText();
        } else if (imagingData.has("TrainingDatasetVersion")) {
            trainingDataset = imagingData.path("TrainingDatasetVersion").asText();
        } else if (imagingData.has("TrainingDataProvenance")) {
            trainingDataset = imagingData.path("TrainingDataProvenance").asText();
        } else if (imagingData.has("TrainingDataset")) {
            trainingDataset = imagingData.path("TrainingDataset").asText();
        } else if (imagingData.has("training_dataset")) {
            trainingDataset = imagingData.path("training_dataset").asText();
        }

        if (trainingDataset == null || trainingDataset.isBlank()) {
            return null;
        }

        Library library = new Library();
        library.setId(UUID.randomUUID().toString());
        library.setName("TrainingData");
        library.setTitle("AI Training Data");
        library.setStatus(org.hl7.fhir.r5.model.Enumerations.PublicationStatus.ACTIVE);
        library.setVersion(trainingDataset);
        library.setType(new CodeableConcept().addCoding(new Coding("http://terminology.hl7.org/CodeSystem/library-type", "asset-collection", "Asset Collection")));
        
        return library;
    }

    /**
     * Creates a Provenance resource capturing the AI inference provenance.
     * - targets: DiagnosticReport (what was created)
     * - agent: AI Device (who/what performed the inference)
     * - entity: ImagingStudy (input data) as derivation
     * - entity: Library (training data) as source/instantiates
     */
    private Provenance createProvenanceResource(JsonNode imagingData, String imagingStudyFullUrl, String aiDeviceFullUrl,
            String deviceFullUrl, String diagnosticReportFullUrl, String trainingLibraryFullUrl) {
        
        // If there is no diagnostic report, we might not need provenance or it targets ImagingStudy?
        // But usually AI creates the report. If only ImagingStudy exists, maybe AI annotated it?
        // For now, assuming AI creates Report.
        if (diagnosticReportFullUrl == null) {
             return null;
        }

        Provenance prov = new Provenance();
        prov.setId(UUID.randomUUID().toString());
        /* Rimosso - il motore FHIR genera automaticamente */
        // prov.setText(new Narrative()
        //         .setStatus(Narrative.NarrativeStatus.GENERATED)
        //         .setDiv(new org.hl7.fhir.r5.model.Narrative.XhtmlNode(org.hl7.fhir.r5.model.NodeType.Element, "div").setValue("Provenance details")));

        // Targets: Only DiagnosticReport (the output of the AI)
        prov.addTarget(new Reference(diagnosticReportFullUrl));

        // Recorded time: use ProcessingTimestamp/InferenceTimestamp or PredictionDateTime if provided, otherwise now
        // Maps to Provenance.occurredDateTime (since R5 allows occurred[x]) or just recorded. 
        // R5 Provenance.recorded is "When the activity was recorded / updated". 
        // Provenance.occurred[x] is "When the activity occurred".
        // ProcessingTimestamp fits occurredDateTime best.
        Date recorded = new Date();
        String timestampStr = extractProcessingTimestamp(imagingData);
        if (timestampStr != null && !timestampStr.isEmpty()) {
            try {
                // Try parsing with RFC patterns similar to above
                java.text.SimpleDateFormat format1 = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX");
                format1.setLenient(false);
                recorded = format1.parse(timestampStr);
                
                // Set occurredDateTime
                prov.setOccurred(new DateTimeType(recorded));
            } catch (Exception e1) {
                 try {
                    java.text.SimpleDateFormat format2 = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
                    format2.setLenient(false);
                    recorded = format2.parse(timestampStr);
                    prov.setOccurred(new DateTimeType(recorded));
                } catch (Exception e2) {
                    // ... (other formats omitted for brevity, fallback to just recorded=now if fail)
                    logger.warn("Could not parse processing timestamp '{}' for occurredDateTime", timestampStr);
                }
            }
        }
        prov.setRecorded(new Date()); // recorded is mandatory, usually "now"

        // Agent: the AI Device as 'performer'
        if (aiDeviceFullUrl != null) {
            ProvenanceAgentComponent agent = new ProvenanceAgentComponent();
            agent.setWho(new Reference(aiDeviceFullUrl));
            // Use "performer" or "author"
            agent.setType(new CodeableConcept().addCoding(new Coding("http://terminology.hl7.org/CodeSystem/provenance-participant-type", "performer", "Performer")));
            prov.addAgent(agent);
        }

        // Entity: ImagingStudy as SOURCE (Input) - DERIVATION removed in R5
        if (imagingStudyFullUrl != null) {
            Provenance.ProvenanceEntityComponent entity = new Provenance.ProvenanceEntityComponent();
            entity.setWhat(new Reference(imagingStudyFullUrl));
            entity.setRole(Provenance.ProvenanceEntityRole.SOURCE);
            prov.addEntity(entity);
        }

        // Entity: Training Data (Library) as source/instantiates
        if (trainingLibraryFullUrl != null) {
            Provenance.ProvenanceEntityComponent entity = new Provenance.ProvenanceEntityComponent();
            entity.setWhat(new Reference(trainingLibraryFullUrl));
            entity.setRole(Provenance.ProvenanceEntityRole.INSTANTIATES); // or SOURCE
            prov.addEntity(entity);
        }

        return prov;
    }
    /**
     * Converts imaging study JSON content to FHIR ImagingStudy Bundle.
     */
    public Bundle convertImagingStudyJsonToFhir(String jsonContent, String fhirVersion) throws Exception {
        if (jsonContent == null || jsonContent.trim().isEmpty()) {
            throw new IllegalArgumentException("Input JSON content is empty or null.");
        }

        JsonNode rootNode = objectMapper.readTree(jsonContent);

        // Validate the JSON structure
        validateJsonStructure(rootNode);

        return createImagingStudyBundle(rootNode, fhirVersion);
    }

    /**
     * Creates a FHIR Bundle containing ImagingStudy and related resources.
     */
    private Bundle createImagingStudyBundle(JsonNode imagingData, String fhirVersion) throws ParseException {
        Bundle bundle = new Bundle();
        bundle.setId(UUID.randomUUID().toString());
        bundle.setType(Bundle.BundleType.COLLECTION);
        bundle.setTimestamp(new Date());

        List<Bundle.BundleEntryComponent> entries = new ArrayList<>();

        // Create Patient resource
        Patient patient = createPatient(imagingData);
        String patientFullUrl = "urn:uuid:" + patient.getId();
        entries.add(createBundleEntry(patientFullUrl, patient));

        // Create ImagingStudy resource
        ImagingStudy imagingStudy = createImagingStudy(imagingData, patientFullUrl);
        String imagingStudyFullUrl = "urn:uuid:" + imagingStudy.getId();
        entries.add(createBundleEntry(imagingStudyFullUrl, imagingStudy));

        // Create Device resource for the imaging equipment
        Device device = createImagingDevice(imagingData);
        if (device != null) {
            String deviceFullUrl = "urn:uuid:" + device.getId();
            entries.add(createBundleEntry(deviceFullUrl, device));
            // Add device reference to the series performer field
            if (!imagingStudy.getSeries().isEmpty()) {
                ImagingStudySeriesComponent series = imagingStudy.getSeries().get(0);
                ImagingStudySeriesPerformerComponent performer = new ImagingStudySeriesPerformerComponent();
                performer.setActor(new Reference(deviceFullUrl));
                series.addPerformer(performer);
            }
        }

        // Create DiagnosticReport for AI classification results
        DiagnosticReport aiDiagnosticReport = createAIDiagnosticReport(imagingData, patientFullUrl,
                imagingStudyFullUrl);
        if (aiDiagnosticReport != null) {
            entries.add(createBundleEntry("urn:uuid:" + aiDiagnosticReport.getId(), aiDiagnosticReport));
        }

        // Create Device resource representing the AI system and add Provenance
        Device aiDevice = createAIDevice(imagingData);
        String aiDeviceFullUrl = null;
        if (aiDevice != null) {
            aiDeviceFullUrl = "urn:uuid:" + aiDevice.getId();
            entries.add(createBundleEntry(aiDeviceFullUrl, aiDevice));
        }

        // Create Library resource for Training Data if available
        Library trainingLibrary = createTrainingDataLibrary(imagingData);
        String trainingLibraryFullUrl = null;
        if (trainingLibrary != null) {
            trainingLibraryFullUrl = "urn:uuid:" + trainingLibrary.getId();
            entries.add(createBundleEntry(trainingLibraryFullUrl, trainingLibrary));
        }

        // Create Provenance resource to track AI inference and link to ImagingStudy, Device and DiagnosticReport
        // Create Provenance resource to track AI inference and link to ImagingStudy, Device and DiagnosticReport
        Provenance provenance = createProvenanceResource(imagingData, imagingStudyFullUrl, aiDeviceFullUrl,
                (device != null) ? "urn:uuid:" + device.getId() : null,
                (aiDiagnosticReport != null) ? "urn:uuid:" + aiDiagnosticReport.getId() : null,
                trainingLibraryFullUrl);
        if (provenance != null) {
            entries.add(createBundleEntry("urn:uuid:" + provenance.getId(), provenance));
        }

        bundle.setEntry(entries);
        return bundle;
    }

    /**
     * Creates Patient resource from DICOM patient information.
     */
    private Patient createPatient(JsonNode imagingData) {
        JsonNode dicomInfo = imagingData.path("DicomInfo");

        Patient patient = new Patient();
        patient.setId(UUID.randomUUID().toString());

        /* Rimosso - il motore FHIR genera automaticamente */
        // patient.setText(new Narrative()
        //         .setStatus(Narrative.NarrativeStatus.GENERATED)
        //         .setDiv(new org.hl7.fhir.r5.model.Narrative.XhtmlNode(org.hl7.fhir.r5.model.NodeType.Element, "div").setValue("Patient details")));

        // Patient ID
        String patientId = dicomInfo.path("PatientID").asText();
        String issuerOfPatientId = dicomInfo.path("IssuerOfPatientID").asText();
        if (!patientId.isEmpty() && !patientId.equals("-sdsd")) {
            Identifier patientIdentifier = patient.addIdentifier()
                    .setValue(patientId);

            // Set system based on issuer if available, otherwise use default
            if (!issuerOfPatientId.isEmpty() && !issuerOfPatientId.equals("N/A")) {
                patientIdentifier.setSystem("urn:issuer:" + issuerOfPatientId);
            } else {
                patientIdentifier.setSystem("urn:dicom:patient-id");
            }
        }

        // Patient Name - prefer separate fields if available, fallback to combined
        // field
        String patientGivenName = dicomInfo.path("PatientGivenName").asText();
        String patientFamilyName = dicomInfo.path("PatientFamilyName").asText();

        if (!patientGivenName.isEmpty() || !patientFamilyName.isEmpty()) {
            // Use separate fields if available
            patient.addName()
                    .setFamily(patientFamilyName)
                    .addGiven(patientGivenName);
        } else {
            // Fallback to combined PatientName field
            String patientNameCombined = dicomInfo.path("PatientName").asText();
            if (!patientNameCombined.isEmpty() && patientNameCombined.contains("^")) {
                String[] nameParts = patientNameCombined.split("\\^");
                if (nameParts.length >= 2) {
                    patient.addName()
                            .setFamily(nameParts[0])
                            .addGiven(nameParts[1]);
                } else {
                    patient.addName().setFamily(patientNameCombined);
                }
            }
        }

        // Birth Date
        String birthDate = dicomInfo.path("PatientBirthDate").asText();
        if (!birthDate.isEmpty() && birthDate.length() == 8) {
            try {
                Date dob = DATE_FORMAT.parse(birthDate);
                patient.setBirthDate(dob);
            } catch (ParseException e) {
                logger.warn("Could not parse birth date: {}", birthDate);
            }
        }

        // Patient Sex/Gender
        String patientSex = dicomInfo.path("PatientSex").asText();
        if (!patientSex.isEmpty() && !patientSex.equals("N/A")) {
            try {
                patient.setGender(mapDicomSexToFhirGender(patientSex));
            } catch (IllegalArgumentException e) {
                logger.warn("Could not map patient sex '{}': {}", patientSex, e.getMessage());
            }
        }

        return patient;
    }

    /**
     * Creates ImagingStudy resource from DICOM study information.
     */
    private ImagingStudy createImagingStudy(JsonNode imagingData, String patientReference) {
        JsonNode dicomInfo = imagingData.path("DicomInfo");

        ImagingStudy imagingStudy = new ImagingStudy();
        imagingStudy.setId(UUID.randomUUID().toString());
        imagingStudy.setStatus(ImagingStudy.ImagingStudyStatus.AVAILABLE);

        /* Rimosso - il motore FHIR genera automaticamente */
        // imagingStudy.setText(new Narrative()
        //         .setStatus(Narrative.NarrativeStatus.GENERATED)
        //         .setDiv(new org.hl7.fhir.r5.model.Narrative.XhtmlNode(org.hl7.fhir.r5.model.NodeType.Element, "div").setValue("Imaging study details")));

        // Subject reference
        imagingStudy.setSubject(new Reference(patientReference));

        // Study Instance UID - use provided UID instead of generating
        String studyInstanceUid = dicomInfo.path("StudyInstanceUID").asText();
        if (!studyInstanceUid.isEmpty() && !studyInstanceUid.equals("N/A")) {
            imagingStudy.addIdentifier()
                    .setSystem("urn:dicom:study-instance-uid")
                    .setValue(studyInstanceUid);
        }

        // Study identifiers
        String accessionNumber = dicomInfo.path("AccessionNumber").asText();
        if (!accessionNumber.isEmpty() && !accessionNumber.equals("250.sdsds")) {
            imagingStudy.addIdentifier()
                    .setSystem("urn:dicom:accession-number")
                    .setValue(accessionNumber);
        }

        // Study ID
        String studyId = dicomInfo.path("StudyID").asText();
        if (!studyId.isEmpty()) {
            imagingStudy.addIdentifier()
                    .setSystem("urn:dicom:study-id")
                    .setValue(studyId);
        }

        // Study description
        String studyDescription = dicomInfo.path("StudyDescription").asText();
        if (!studyDescription.isEmpty()) {
            imagingStudy.setDescription(studyDescription);
        }

        // Modality
        String modality = dicomInfo.path("Modality").asText();
        if (!modality.isEmpty()) {
            imagingStudy.setModality(List.of(new CodeableConcept()
                    .addCoding(new Coding()
                            .setSystem("http://dicom.nema.org/resources/ontology/DCM")
                            .setCode(modality)
                            .setDisplay(getModalityDisplay(modality)))));
        }

        // Started date/time - prefer AcquisitionDateTime if available, fallback to
        // AcquisitionDate
        String acquisitionDateTime = dicomInfo.path("AcquisitionDateTime").asText();
        if (!acquisitionDateTime.isEmpty()) {
            try {
                // Parse ISO datetime format - try different formats with/without timezone and
                // microseconds/milliseconds
                Date startDate = null;
                ParseException lastException = null;

                // Try with microseconds and timezone (with colon like +02:00)
                try {
                    SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX");
                    format1.setLenient(false);
                    startDate = format1.parse(acquisitionDateTime);
                } catch (ParseException e1) {
                    lastException = e1;
                    // Try with microseconds and timezone (without colon like +0200)
                    try {
                        SimpleDateFormat format1b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZ");
                        format1b.setLenient(false);
                        startDate = format1b.parse(acquisitionDateTime);
                    } catch (ParseException e1b) {
                        lastException = e1b;
                        // Try with microseconds without timezone
                        try {
                            SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                            format2.setLenient(false);
                            startDate = format2.parse(acquisitionDateTime);
                        } catch (ParseException e2) {
                            lastException = e2;
                            // Try with milliseconds and timezone (with colon like +02:00)
                            try {
                                SimpleDateFormat format3 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
                                format3.setLenient(false);
                                startDate = format3.parse(acquisitionDateTime);
                            } catch (ParseException e3) {
                                lastException = e3;
                                // Try with milliseconds and timezone (without colon like +0200)
                                try {
                                    SimpleDateFormat format3b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZZ");
                                    format3b.setLenient(false);
                                    startDate = format3b.parse(acquisitionDateTime);
                                } catch (ParseException e3b) {
                                    lastException = e3b;
                                    // Try with milliseconds without timezone
                                    try {
                                        SimpleDateFormat format4 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS");
                                        format4.setLenient(false);
                                        startDate = format4.parse(acquisitionDateTime);
                                    } catch (ParseException e4) {
                                        lastException = e4;
                                        logger.warn("Could not parse acquisition datetime '{}': {}",
                                                acquisitionDateTime,
                                                lastException.getMessage());
                                    }
                                }
                            }
                        }
                    }
                }

                if (startDate != null) {
                    imagingStudy.setStarted(startDate);
                }
            } catch (Exception e) {
                logger.warn("Could not parse acquisition datetime: {}", acquisitionDateTime);
                // Fallback to AcquisitionDate
                String acquisitionDate = dicomInfo.path("AcquisitionDate").asText();
                if (!acquisitionDate.isEmpty() && acquisitionDate.length() == 8) {
                    try {
                        Date fallbackDate = DATE_FORMAT.parse(acquisitionDate);
                        imagingStudy.setStarted(fallbackDate);
                    } catch (ParseException e2) {
                        logger.warn("Could not parse acquisition date: {}", acquisitionDate);
                    }
                }
            }
        } else {
            // Fallback to AcquisitionDate
            String acquisitionDate = dicomInfo.path("AcquisitionDate").asText();
            if (!acquisitionDate.isEmpty() && acquisitionDate.length() == 8) {
                try {
                    Date startDate = DATE_FORMAT.parse(acquisitionDate);
                    imagingStudy.setStarted(startDate);
                } catch (ParseException e) {
                    logger.warn("Could not parse acquisition date: {}", acquisitionDate);
                }
            }
        }

        // Create series
        ImagingStudySeriesComponent series = createImagingSeries(imagingData);
        if (series != null) {
            imagingStudy.addSeries(series);
        }

        return imagingStudy;
    }

    /**
     * Creates imaging series from the data.
     */
    private ImagingStudySeriesComponent createImagingSeries(JsonNode imagingData) {
        JsonNode dicomInfo = imagingData.path("DicomInfo");
        JsonNode imageParams = dicomInfo.path("ImageParameters");

        ImagingStudySeriesComponent series = new ImagingStudySeriesComponent();
        series.setId(UUID.randomUUID().toString());

        // Use provided SeriesInstanceUID if available, otherwise generate
        String seriesInstanceUid = dicomInfo.path("SeriesInstanceUID").asText();
        if (!seriesInstanceUid.isEmpty() && !seriesInstanceUid.equals("N/A")) {
            series.setUid(seriesInstanceUid);
        } else {
            series.setUid(generateDicomUid()); // Generate if not provided
        }

        series.setNumber(1); // Series number

        // Modality
        String modality = dicomInfo.path("Modality").asText();
        if (!modality.isEmpty()) {
            series.setModality(new CodeableConcept()
                    .addCoding(new Coding()
                            .setSystem("http://dicom.nema.org/resources/ontology/DCM")
                            .setCode(modality)
                            .setDisplay(getModalityDisplay(modality))));
        }

        // Series description
        String seriesDescription = dicomInfo.path("SeriesDescription").asText();
        if (!seriesDescription.isEmpty() && !seriesDescription.equals("N/A")) {
            series.setDescription(seriesDescription);
        }

        // Number of instances
        int numberOfInstances = dicomInfo.path("numberOfStudyRelatedInstances").asInt(1);
        series.setNumberOfInstances(numberOfInstances);

        // Create instance
        ImagingStudySeriesInstanceComponent instance = createImagingInstance(imagingData);
        if (instance != null) {
            series.addInstance(instance);
        }

        return series;
    }

    /**
     * Creates imaging instance from the data.
     */
    private ImagingStudySeriesInstanceComponent createImagingInstance(JsonNode imagingData) {
        JsonNode dicomInfo = imagingData.path("DicomInfo");
        JsonNode imageParams = dicomInfo.path("ImageParameters");

        ImagingStudySeriesInstanceComponent instance = new ImagingStudySeriesInstanceComponent();
        instance.setId(UUID.randomUUID().toString());

        // Use provided SOPInstanceUID if available, otherwise generate
        String sopInstanceUid = dicomInfo.path("SOPInstanceUID").asText();
        if (!sopInstanceUid.isEmpty() && !sopInstanceUid.equals("N/A")) {
            instance.setUid(sopInstanceUid);
        } else {
            instance.setUid(generateDicomUid()); // Generate if not provided
        }

        instance.setNumber(1); // Instance number

        // Image type - assume original for now
        instance.setTitle("Original Image");

        // Add sopClass - required field for FHIR compliance
        // Use DICOM modality code with correct CodeSystem URI
        String modality = dicomInfo.path("Modality").asText();
        instance.setSopClass(new Coding()
                .setSystem("urn:ietf:rfc:3986")
                .setCode("urn:oid:" + getSOPClassCode(modality))
                .setDisplay(getSOPClassDisplay(modality))); // Use modality display name

        return instance;
    }

    /**
     * Creates Device resource for imaging equipment.
     */
    private Device createImagingDevice(JsonNode imagingData) {
        JsonNode dicomInfo = imagingData.path("DicomInfo");

        String manufacturer = dicomInfo.path("Manufacturer").asText();
        String modelName = dicomInfo.path("ManufacturerModelName").asText();
        String deviceSerial = dicomInfo.path("DeviceSerialNumber").asText();
        String softwareVersion = dicomInfo.path("SoftwareVersions").asText();
        String implementationVersion = dicomInfo.path("ImplementationVersionName").asText();
        String spatialResolution = dicomInfo.path("SpatialResolution").asText();
        String detectorModel = dicomInfo.path("DetectorManufacturerModelName").asText();

        if (manufacturer.isEmpty() && modelName.isEmpty() && deviceSerial.isEmpty()) {
            return null; // No device information available
        }

        Device device = new Device();
        device.setId(UUID.randomUUID().toString());
        device.setStatus(Device.FHIRDeviceStatus.ACTIVE);

        /* Rimosso - il motore FHIR genera automaticamente */
        // device.setText(new Narrative()
        //         .setStatus(Narrative.NarrativeStatus.GENERATED)
        //         .setDiv(new org.hl7.fhir.r5.model.Narrative.XhtmlNode(org.hl7.fhir.r5.model.NodeType.Element, "div").setValue("Imaging device details")));

        // Manufacturer
        if (!manufacturer.isEmpty() && !manufacturer.equals("N/A")) {
            device.setManufacturer(manufacturer);
        }

        // Model name
        if (!modelName.isEmpty() && !modelName.equals("N/A")) {
            device.setModelNumber(modelName);
        }

        // Serial number
        if (!deviceSerial.isEmpty() && !deviceSerial.equals("N/A")) {
            device.addIdentifier()
                    .setSystem("urn:dicom:device-serial")
                    .setValue(deviceSerial);
        }

        // Software version - combine SoftwareVersions and ImplementationVersionName if
        // available
        List<Device.DeviceVersionComponent> versions = new ArrayList<>();
        if (!softwareVersion.isEmpty() && !softwareVersion.equals("N/A")) {
            versions.add(new Device.DeviceVersionComponent().setValue(softwareVersion));
        }
        if (!implementationVersion.isEmpty() && !implementationVersion.equals("N/A")) {
            versions.add(new Device.DeviceVersionComponent().setValue(implementationVersion));
        }
        if (!versions.isEmpty()) {
            device.setVersion(versions);
        }

        // Add spatial resolution and detector model information as device properties
        if (!spatialResolution.isEmpty() && !spatialResolution.equals("N/A")) {
            device.setType(List.of(new CodeableConcept()
                    .addCoding(new Coding()
                            .setSystem("http://dicom.nema.org/resources/ontology/DCM")
                            .setCode("108800")
                            .setDisplay("Digital X-ray system"))));
        }

        if (!detectorModel.isEmpty() && !detectorModel.equals("N/A")) {
            // Use device identifier for detector model
            device.addIdentifier()
                    .setSystem("urn:dicom:detector-model")
                    .setValue(detectorModel);
        }

        // Device type - use a valid SNOMED CT code for patient health record
        // information system
        device.setType(List.of(new CodeableConcept()
                .addCoding(new Coding()
                        .setSystem("http://snomed.info/sct")
                        .setCode("462240000") // Patient health record information system
                        .setDisplay("Patient health record information system"))));

        return device;
    }

    /**
     * Creates DiagnosticReport for AI classification results.
     * Processes the new JSON structure with ClassificationEncoding and
     * PredictionProbabilities.
     */
    private DiagnosticReport createAIDiagnosticReport(JsonNode imagingData, String patientReference,
            String imagingStudyReference) {
        // Validate required fields
        if (!imagingData.has("Classification") || !imagingData.has("PredictionAccuracy")) {
            logger.warn("Missing required AI classification fields");
            return null;
        }

        String classification = imagingData.path("Classification").asText();
        double accuracy = imagingData.path("PredictionAccuracy").asDouble();

        if (classification.isEmpty() || accuracy == 0.0) {
            logger.warn("Invalid AI classification data - empty classification or zero accuracy");
            return null; // No AI results available
        }

        DiagnosticReport diagnosticReport = new DiagnosticReport();
        diagnosticReport.setId(UUID.randomUUID().toString());
        diagnosticReport.setStatus(DiagnosticReportStatus.FINAL);

        /* Rimosso - il motore FHIR genera automaticamente */
        // diagnosticReport.setText(new Narrative()
        //         .setStatus(Narrative.NarrativeStatus.GENERATED)
        //         .setDiv(new org.hl7.fhir.r5.model.Narrative.XhtmlNode(org.hl7.fhir.r5.model.NodeType.Element, "div").setValue("AI analysis diagnostic report")));

        // Subject reference
        diagnosticReport.setSubject(new Reference(patientReference));

        // ImagingStudy reference - link to the study that was analyzed
        if (imagingStudyReference != null && !imagingStudyReference.isEmpty()) {
            diagnosticReport.setStudy(List.of(new Reference(imagingStudyReference)));
        }

        // DiagnosticReport code - use a more appropriate LOINC code
        diagnosticReport.setCode(new CodeableConcept()
                .addCoding(new Coding()
                        .setSystem("http://loinc.org")
                        .setCode("36643-5") // XR Chest 2 Views
                        .setDisplay("XR Chest 2V")));

        // Conclusion - human-readable summary
        String conclusion = getConclusionText(classification);
        diagnosticReport.setConclusion(conclusion);

        // Conclusion code - SNOMED CT code
        CodeableConcept conclusionCode = new CodeableConcept()
                .addCoding(new Coding()
                        .setSystem("http://snomed.info/sct")
                        .setCode(getClassificationCode(classification))
                        .setDisplay(getClassificationDisplay(classification)));
        diagnosticReport.setConclusionCode(List.of(conclusionCode));

        // Process PredictionDateTime if available
        if (imagingData.has("PredictionDateTime")) {
            String predictionDateTime = imagingData.path("PredictionDateTime").asText();
            if (!predictionDateTime.isEmpty()) {
                try {
                    // Parse ISO 8601 datetime format - try different formats with/without timezone
                    // and microseconds/milliseconds
                    Date issuedDate = null;
                    ParseException lastException = null;

                    // Try with microseconds and timezone (with colon like +02:00)
                    try {
                        SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX");
                        format1.setLenient(false);
                        issuedDate = format1.parse(predictionDateTime);
                    } catch (ParseException e1) {
                        lastException = e1;
                        // Try with microseconds and timezone (without colon like +0200)
                        try {
                            SimpleDateFormat format1b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZ");
                            format1b.setLenient(false);
                            issuedDate = format1b.parse(predictionDateTime);
                        } catch (ParseException e1b) {
                            lastException = e1b;
                            // Try with microseconds without timezone
                            try {
                                SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                                format2.setLenient(false);
                                issuedDate = format2.parse(predictionDateTime);
                            } catch (ParseException e2) {
                                lastException = e2;
                                // Try with milliseconds and timezone (with colon like +02:00)
                                try {
                                    SimpleDateFormat format3 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
                                    format3.setLenient(false);
                                    issuedDate = format3.parse(predictionDateTime);
                                } catch (ParseException e3) {
                                    lastException = e3;
                                    // Try with milliseconds and timezone (without colon like +0200)
                                    try {
                                        SimpleDateFormat format3b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZZ");
                                        format3b.setLenient(false);
                                        issuedDate = format3b.parse(predictionDateTime);
                                    } catch (ParseException e3b) {
                                        lastException = e3b;
                                        // Try with milliseconds without timezone
                                        try {
                                            SimpleDateFormat format4 = new SimpleDateFormat(
                                                    "yyyy-MM-dd'T'HH:mm:ss.SSS");
                                            format4.setLenient(false);
                                            issuedDate = format4.parse(predictionDateTime);
                                        } catch (ParseException e4) {
                                            lastException = e4;
                                            // Try with seconds and timezone (with colon like +02:00)
                                            try {
                                                SimpleDateFormat format5 = new SimpleDateFormat(
                                                        "yyyy-MM-dd'T'HH:mm:ssXXX");
                                                format5.setLenient(false);
                                                issuedDate = format5.parse(predictionDateTime);
                                            } catch (ParseException e5) {
                                                lastException = e5;
                                                // Try with seconds and timezone (without colon like +0200)
                                                try {
                                                    SimpleDateFormat format5b = new SimpleDateFormat(
                                                            "yyyy-MM-dd'T'HH:mm:ssZZ");
                                                    format5b.setLenient(false);
                                                    issuedDate = format5b.parse(predictionDateTime);
                                                } catch (ParseException e5b) {
                                                    lastException = e5b;
                                                    // Try with seconds without timezone
                                                    try {
                                                        SimpleDateFormat format6 = new SimpleDateFormat(
                                                                "yyyy-MM-dd'T'HH:mm:ss");
                                                        format6.setLenient(false);
                                                        issuedDate = format6.parse(predictionDateTime);
                                                    } catch (ParseException e6) {
                                                        lastException = e6;
                                                        logger.warn("Could not parse PredictionDateTime '{}': {}",
                                                                predictionDateTime,
                                                                lastException.getMessage());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (issuedDate != null) {
                        diagnosticReport.setIssued(issuedDate);
                    }
                } catch (Exception e) {
                    logger.warn("Could not parse PredictionDateTime: {}", predictionDateTime);
                    // Fall back to current date
                    diagnosticReport.setIssued(new Date());
                }
            } else {
                diagnosticReport.setIssued(new Date());
            }
        } else {
            // Issued timestamp
            diagnosticReport.setIssued(new Date());

            // Process ClassificationEncoding if available
            if (imagingData.has("ClassificationEncoding")) {
                JsonNode classificationEncoding = imagingData.path("ClassificationEncoding");
                if (classificationEncoding.has("ClassificationMap")) {
                    Extension classificationMapExtension = new Extension();
                    classificationMapExtension
                            .setUrl("http://my-hospital.org/fhir/StructureDefinition/classification-map");
                    classificationEncoding.path("ClassificationMap").fields().forEachRemaining(entry -> {
                        Extension mapEntry = new Extension();
                        mapEntry.setUrl("http://my-hospital.org/fhir/StructureDefinition/classification-map-entry");
                        mapEntry.addExtension("key", new StringType(entry.getKey()));
                        mapEntry.addExtension("value", new StringType(entry.getValue().asText()));
                        classificationMapExtension.addExtension(mapEntry);
                    });
                    diagnosticReport.addExtension(classificationMapExtension);
                }
            }

            // Process PredictionProbabilities if available
            if (imagingData.has("PredictionProbabilities")) {
                JsonNode probabilities = imagingData.path("PredictionProbabilities");
                if (probabilities.isArray()) {
                    Extension probabilitiesExtension = new Extension();
                    probabilitiesExtension
                            .setUrl("http://my-hospital.org/fhir/StructureDefinition/prediction-probabilities");
                    probabilities.forEach(prob -> {
                        probabilitiesExtension.addExtension("probability", new DecimalType(prob.asDouble()));
                    });
                    diagnosticReport.addExtension(probabilitiesExtension);
                }
            }
        }

        return diagnosticReport;
    }

    /**
     * Gets human-readable conclusion text for classification results.
     */
    private String getConclusionText(String classification) {
        switch (classification.toUpperCase()) {
            case "NORMAL":
                return "No acute abnormality detected.";
            case "AORTIC_CALCIFICATION":
                return "Aortic calcification detected.";
            default:
                return "Findings require clinical correlation.";
        }
    }

    /**
     * Creates a bundle entry for a resource.
     */
    private Bundle.BundleEntryComponent createBundleEntry(String fullUrl, Resource resource) {
        return new Bundle.BundleEntryComponent()
                .setFullUrl(fullUrl)
                .setResource(resource);
    }

    /**
     * Gets display name for DICOM modality codes.
     */
    private String getModalityDisplay(String modality) {
        switch (modality.toUpperCase()) {
            case "DX":
                return "Digital Radiography";
            case "CT":
                return "Computed Tomography";
            case "MR":
                return "Magnetic Resonance";
            case "US":
                return "Ultrasound";
            case "CR":
                return "Computed Radiography";
            case "RF":
                return "Radiofluoroscopy";
            default:
                return "Unknown Modality";
        }
    }

    /**
     * Gets SOP Class code for DICOM modalities.
     */
    private String getSOPClassCode(String modality) {
        switch (modality.toUpperCase()) {
            case "DX":
                return "1.2.840.10008.5.1.4.1.1.1.1"; // Digital X-Ray Image Storage
            case "CT":
                return "1.2.840.10008.5.1.4.1.1.2"; // CT Image Storage
            case "MR":
                return "1.2.840.10008.5.1.4.1.1.4"; // MR Image Storage
            case "US":
                return "1.2.840.10008.5.1.4.1.1.6"; // Ultrasound Image Storage
            case "CR":
                return "1.2.840.10008.5.1.4.1.1.1"; // Computed Radiography Image Storage
            default:
                return DICOM_SOP_CLASS_UID;
        }
    }

    /**
     * Gets SOP Class display for DICOM modalities.
     */
    private String getSOPClassDisplay(String modality) {
        switch (modality.toUpperCase()) {
            case "DX":
                return "Digital X-Ray Image Storage";
            case "CT":
                return "CT Image Storage";
            case "MR":
                return "MR Image Storage";
            case "US":
                return "Ultrasound Image Storage";
            case "CR":
                return "Computed Radiography Image Storage";
            default:
                return "Image Storage";
        }
    }

    /**
     * Gets SNOMED CT code for classification results.
     */
    private String getClassificationCode(String classification) {
        switch (classification.toUpperCase()) {
            case "NORMAL":
                return "17621005"; // Normal
            case "AORTIC_CALCIFICATION":
                return "155029007"; // Aortic calcification
            default:
                return "261665006"; // Unknown
        }
    }

    /**
     * Gets SNOMED CT display for classification results.
     */
    private String getClassificationDisplay(String classification) {
        switch (classification.toUpperCase()) {
            case "NORMAL":
                return "Normal";
            case "AORTIC_CALCIFICATION":
                return "Aortic calcification";
            default:
                return "Unknown finding";
        }
    }

    /**
     * Maps DICOM PatientSex values to FHIR AdministrativeGender.
     */
    private Enumerations.AdministrativeGender mapDicomSexToFhirGender(String dicomSex) {
        if (dicomSex == null || dicomSex.trim().isEmpty()) {
            throw new IllegalArgumentException("DICOM sex value cannot be null or empty");
        }

        switch (dicomSex.toUpperCase().trim()) {
            case "M":
                return Enumerations.AdministrativeGender.MALE;
            case "F":
                return Enumerations.AdministrativeGender.FEMALE;
            case "O":
                return Enumerations.AdministrativeGender.OTHER;
            default:
                throw new IllegalArgumentException("Unknown DICOM sex value: " + dicomSex + ". Expected M, F, or O");
        }
    }

    /**
     * Generates a proper DICOM UID using the organization root.
     */
    private String generateDicomUid() {
        long timestamp = System.currentTimeMillis();
        long random = (long) (Math.random() * 1000000L);
        return DICOM_ORG_ROOT + "." + timestamp + "." + random;
    }

    /**
     * Converts imaging study JSON content to FHIR JSON string.
     */
    public String convertToFhirJson(String jsonContent, String fhirVersion) throws Exception {
        Bundle bundle = convertImagingStudyJsonToFhir(jsonContent, fhirVersion);
        return fhirContext.newJsonParser().setPrettyPrint(true).encodeResourceToString(bundle);
    }

    /**
     * Converts imaging study JSON file to FHIR JSON file.
     */
    public void convertFile(String inputPath, String outputPath, String fhirVersion) throws Exception {
        String jsonContent = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(inputPath)));
        String fhirJson = convertToFhirJson(jsonContent, fhirVersion);
        java.nio.file.Files.writeString(java.nio.file.Paths.get(outputPath), fhirJson);
    }

    /**
     * Validates the JSON structure to ensure data integrity.
     * Checks for required fields and proper data types.
     */
    private void validateJsonStructure(JsonNode rootNode) throws IllegalArgumentException {
        // Check for required top-level fields
        if (!rootNode.has("Classification")) {
            throw new IllegalArgumentException("Missing required field: Classification");
        }
        if (!rootNode.has("PredictionAccuracy")) {
            throw new IllegalArgumentException("Missing required field: PredictionAccuracy");
        }
        if (!rootNode.has("DicomInfo")) {
            throw new IllegalArgumentException("Missing required field: DicomInfo");
        }

        // Validate Classification field
        String classification = rootNode.path("Classification").asText();
        if (classification.isEmpty()) {
            throw new IllegalArgumentException("Classification field cannot be empty");
        }

        // Validate PredictionAccuracy field
        double accuracy = rootNode.path("PredictionAccuracy").asDouble();
        if (accuracy < 0.0 || accuracy > 1.0) {
            throw new IllegalArgumentException("PredictionAccuracy must be between 0.0 and 1.0");
        }

        // Validate ClassificationEncoding if present
        if (rootNode.has("ClassificationEncoding")) {
            JsonNode encoding = rootNode.path("ClassificationEncoding");
            if (encoding.has("ClassificationMap")) {
                JsonNode classificationMap = encoding.path("ClassificationMap");
                if (!classificationMap.isObject()) {
                    throw new IllegalArgumentException("ClassificationMap must be an object");
                }
            }
        }

        // Validate PredictionProbabilities if present
        if (rootNode.has("PredictionProbabilities")) {
            JsonNode probabilities = rootNode.path("PredictionProbabilities");
            if (!probabilities.isArray()) {
                throw new IllegalArgumentException("PredictionProbabilities must be an array");
            }
            for (JsonNode prob : probabilities) {
                double value = prob.asDouble();
                if (value < 0.0 || value > 1.0) {
                    throw new IllegalArgumentException("All prediction probabilities must be between 0.0 and 1.0");
                }
            }
        }

        // Validate PredictionDateTime if present
        if (rootNode.has("PredictionDateTime")) {
            String dateTime = rootNode.path("PredictionDateTime").asText();
            if (!dateTime.isEmpty()) {
                try {
                    // Try different formats: with/without timezone, microseconds/milliseconds
                    Date parsedDate = null;
                    ParseException lastException = null;

                    // Try with microseconds and timezone (with colon like +02:00)
                    try {
                        SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX");
                        format1.setLenient(false);
                        parsedDate = format1.parse(dateTime);
                    } catch (ParseException e1) {
                        lastException = e1;
                        // Try with microseconds and timezone (without colon like +0200)
                        try {
                            SimpleDateFormat format1b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZ");
                            format1b.setLenient(false);
                            parsedDate = format1b.parse(dateTime);
                        } catch (ParseException e1b) {
                            lastException = e1b;
                            // Try with microseconds without timezone
                            try {
                                SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                                format2.setLenient(false);
                                parsedDate = format2.parse(dateTime);
                            } catch (ParseException e2) {
                                lastException = e2;
                                // Try with milliseconds and timezone (with colon like +02:00)
                                try {
                                    SimpleDateFormat format3 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
                                    format3.setLenient(false);
                                    parsedDate = format3.parse(dateTime);
                                } catch (ParseException e3) {
                                    lastException = e3;
                                    // Try with milliseconds and timezone (without colon like +0200)
                                    try {
                                        SimpleDateFormat format3b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZZ");
                                        format3b.setLenient(false);
                                        parsedDate = format3b.parse(dateTime);
                                    } catch (ParseException e3b) {
                                        lastException = e3b;
                                        // Try with milliseconds without timezone
                                        try {
                                            SimpleDateFormat format4 = new SimpleDateFormat(
                                                    "yyyy-MM-dd'T'HH:mm:ss.SSS");
                                            format4.setLenient(false);
                                            parsedDate = format4.parse(dateTime);
                                        } catch (ParseException e4) {
                                            lastException = e4;
                                            // Try with seconds and timezone (with colon like +02:00)
                                            try {
                                                SimpleDateFormat format5 = new SimpleDateFormat(
                                                        "yyyy-MM-dd'T'HH:mm:ssXXX");
                                                format5.setLenient(false);
                                                parsedDate = format5.parse(dateTime);
                                            } catch (ParseException e5) {
                                                lastException = e5;
                                                // Try with seconds and timezone (without colon like +0200)
                                                try {
                                                    SimpleDateFormat format5b = new SimpleDateFormat(
                                                            "yyyy-MM-dd'T'HH:mm:ssZZ");
                                                    format5b.setLenient(false);
                                                    parsedDate = format5b.parse(dateTime);
                                                } catch (ParseException e5b) {
                                                    lastException = e5b;
                                                    // Try with seconds without timezone
                                                    try {
                                                        SimpleDateFormat format6 = new SimpleDateFormat(
                                                                "yyyy-MM-dd'T'HH:mm:ss");
                                                        format6.setLenient(false);
                                                        parsedDate = format6.parse(dateTime);
                                                    } catch (ParseException e6) {
                                                        lastException = e6;
                                                        throw new IllegalArgumentException(
                                                                "Invalid PredictionDateTime format. Expected ISO 8601 format: yyyy-MM-dd'T'HH:mm:ss.SSS or yyyy-MM-dd'T'HH:mm:ss.SSSSSS, with or without timezone",
                                                                lastException);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } catch (IllegalArgumentException e) {
                    throw e;
                } catch (Exception e) {
                    throw new IllegalArgumentException(
                            "Invalid PredictionDateTime format. Expected ISO 8601 format: yyyy-MM-dd'T'HH:mm:ss.SSS or yyyy-MM-dd'T'HH:mm:ss.SSSSSS, with or without timezone",
                            e);
                }
            }
        }

        // Validate DicomInfo structure
        JsonNode dicomInfo = rootNode.path("DicomInfo");
        if (dicomInfo.isMissingNode() || !dicomInfo.isObject()) {
            throw new IllegalArgumentException("DicomInfo must be an object");
        }

        // Validate required DicomInfo fields
        String patientId = dicomInfo.path("PatientID").asText();
        if (patientId.isEmpty()) {
            throw new IllegalArgumentException("DicomInfo.PatientID cannot be empty");
        }

        String patientName = dicomInfo.path("PatientName").asText();
        String patientGivenName = dicomInfo.path("PatientGivenName").asText();
        String patientFamilyName = dicomInfo.path("PatientFamilyName").asText();

        if (patientName.isEmpty() && patientGivenName.isEmpty() && patientFamilyName.isEmpty()) {
            throw new IllegalArgumentException(
                    "Patient name information is required (PatientName or PatientGivenName + PatientFamilyName)");
        }

        // Validate date formats in DicomInfo
        String patientBirthDate = dicomInfo.path("PatientBirthDate").asText();
        if (!patientBirthDate.isEmpty() && patientBirthDate.length() != 8) {
            throw new IllegalArgumentException("PatientBirthDate must be in format yyyyMMdd");
        }

        String acquisitionDateTime = dicomInfo.path("AcquisitionDateTime").asText();
        if (!acquisitionDateTime.isEmpty()) {
            try {
                // Try different formats: with/without timezone, microseconds/milliseconds
                Date parsedDate = null;
                ParseException lastException = null;

                // Try with microseconds and timezone (with colon like +02:00)
                try {
                    SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX");
                    format1.setLenient(false);
                    parsedDate = format1.parse(acquisitionDateTime);
                } catch (ParseException e1) {
                    lastException = e1;
                    // Try with microseconds and timezone (without colon like +0200)
                    try {
                        SimpleDateFormat format1b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZ");
                        format1b.setLenient(false);
                        parsedDate = format1b.parse(acquisitionDateTime);
                    } catch (ParseException e1b) {
                        lastException = e1b;
                        // Try with microseconds without timezone
                        try {
                            SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                            format2.setLenient(false);
                            parsedDate = format2.parse(acquisitionDateTime);
                        } catch (ParseException e2) {
                            lastException = e2;
                            // Try with milliseconds and timezone (with colon like +02:00)
                            try {
                                SimpleDateFormat format3 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
                                format3.setLenient(false);
                                parsedDate = format3.parse(acquisitionDateTime);
                            } catch (ParseException e3) {
                                lastException = e3;
                                // Try with milliseconds and timezone (without colon like +0200)
                                try {
                                    SimpleDateFormat format3b = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZZ");
                                    format3b.setLenient(false);
                                    parsedDate = format3b.parse(acquisitionDateTime);
                                } catch (ParseException e3b) {
                                    lastException = e3b;
                                    // Try with milliseconds without timezone
                                    try {
                                        SimpleDateFormat format4 = new SimpleDateFormat(
                                                "yyyy-MM-dd'T'HH:mm:ss.SSS");
                                        format4.setLenient(false);
                                        parsedDate = format4.parse(acquisitionDateTime);
                                    } catch (ParseException e4) {
                                        lastException = e4;
                                        // Try with seconds and timezone (with colon like +02:00)
                                        try {
                                            SimpleDateFormat format5 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX");
                                            format5.setLenient(false);
                                            parsedDate = format5.parse(acquisitionDateTime);
                                        } catch (ParseException e5) {
                                            lastException = e5;
                                            // Try with seconds and timezone (without colon like +0200)
                                            try {
                                                SimpleDateFormat format5b = new SimpleDateFormat(
                                                        "yyyy-MM-dd'T'HH:mm:ssZZ");
                                                format5b.setLenient(false);
                                                parsedDate = format5b.parse(acquisitionDateTime);
                                            } catch (ParseException e5b) {
                                                lastException = e5b;
                                                // Try with seconds without timezone
                                                try {
                                                    SimpleDateFormat format6 = new SimpleDateFormat(
                                                            "yyyy-MM-dd'T'HH:mm:ss");
                                                    format6.setLenient(false);
                                                    parsedDate = format6.parse(acquisitionDateTime);
                                                } catch (ParseException e6) {
                                                    lastException = e6;
                                                    throw new IllegalArgumentException(
                                                            "Invalid AcquisitionDateTime format. Expected ISO 8601 format: yyyy-MM-dd'T'HH:mm:ss.SSS or yyyy-MM-dd'T'HH:mm:ss.SSSSSS, with or without timezone",
                                                            lastException);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (IllegalArgumentException e) {
                throw e;
            } catch (Exception e) {
                throw new IllegalArgumentException(
                        "Invalid AcquisitionDateTime format. Expected ISO 8601 format: yyyy-MM-dd'T'HH:mm:ss.SSS or yyyy-MM-dd'T'HH:mm:ss.SSSSSS, with or without timezone",
                        e);
            }
        }

        // Validate UIDs if present
        String studyInstanceUid = dicomInfo.path("StudyInstanceUID").asText();
        if (!studyInstanceUid.isEmpty() && !studyInstanceUid.equals("N/A")) {
            if (!studyInstanceUid.matches("[0-9.]+")) {
                throw new IllegalArgumentException("StudyInstanceUID must contain only digits and dots");
            }
        }

        String seriesInstanceUid = dicomInfo.path("SeriesInstanceUID").asText();
        if (!seriesInstanceUid.isEmpty() && !seriesInstanceUid.equals("N/A")) {
            if (!seriesInstanceUid.matches("[0-9.]+")) {
                throw new IllegalArgumentException("SeriesInstanceUID must contain only digits and dots");
            }
        }

        String sopInstanceUid = dicomInfo.path("SOPInstanceUID").asText();
        if (!sopInstanceUid.isEmpty() && !sopInstanceUid.equals("N/A")) {
            if (!sopInstanceUid.matches("[0-9.]+")) {
                throw new IllegalArgumentException("SOPInstanceUID must contain only digits and dots");
            }
        }

        logger.info("JSON validation passed successfully");
    }
}
