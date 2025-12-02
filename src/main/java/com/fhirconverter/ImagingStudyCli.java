package com.fhirconverter;

public class ImagingStudyCli {

    public static void main(String[] args) {
        if (args == null || args.length < 2) {
            System.err.println("Usage: ImagingStudyCli <inputJsonFile> <outputFhirJsonFile> [fhirVersion]");
            System.exit(1);
        }

        String input = args[0];
        String output = args[1];
        String fhirVersion = (args.length >= 3 && args[2] != null && !args[2].trim().isEmpty()) ? args[2] : "R5";

        try {
            ImagingStudyConverter converter = new ImagingStudyConverter();
            converter.convertFile(input, output, fhirVersion);
            System.out.println("FHIR Bundle generated to: " + output);
        } catch (Exception e) {
            System.err.println("Error during conversion: " + e.getMessage());
            e.printStackTrace();
            System.exit(2);
        }
    }
}
