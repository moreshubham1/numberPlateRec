#include <iostream>
#include <string>
#include <codecvt>
#include <locale>
#include "SimpleLPR.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // Convert char* to wchar_t* using std::wstring_convert
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wide_image_path = converter.from_bytes(argv[1]);
    const wchar_t* imagePath = wide_image_path.c_str();

    // Initialize the SimpleLPR engine setup parameters
    SIMPLELPR_Engine_Setup_Parms setupParams;
    setupParams.cudaDeviceId = -1;  // Use CPU
    setupParams.enableImageProcessingWithGPU = 0;  // Do not use GPU
    setupParams.enableClassificationWithGPU = 0;  // Do not use GPU
    setupParams.maxConcurrentImageProcessingOps = 0;  // Use default
    setupParams.guard = SIMPLELPR_SETUP_PARMS_GUARD;

    // Create the SimpleLPR engine
    SIMPLELPR_Handle engine = SIMPLELPR_Setup(&setupParams);
    if (!engine) {
        std::cerr << "Error: Failed to initialize SimpleLPR engine." << std::endl;
        return 1;
    }
    std::cout << "Engine found!!!!" << std::endl;

    std::cout << "Create a processor" << std::endl;
    SIMPLELPR_Handle processor = SIMPLELPR_createProcessor(engine);
    if (!processor) {
        std::cerr << "Error: Failed to create processor." << std::endl;
        SIMPLELPR_ReferenceCounted_release(engine);
        return 1;
    }

    // Analyze the image file
    std::cout << "Analyze the image file" << std::endl;
    SIMPLELPR_Handle candidates = SIMPLELPR_Processor_analyzeFile(processor, imagePath);
    if (!candidates) {
        SIMPLELPR_Handle errorInfo = SIMPLELPR_lastError_get(engine, 0);  // 0 instead of SIMPLELPR_BOOL_FALSE
        if (errorInfo) {
            const wchar_t* errorMessage = SIMPLELPR_ErrorInfo_description_get(errorInfo);  // Use description_get
            std::wcerr << L"Error: " << errorMessage << std::endl;
            SIMPLELPR_ReferenceCounted_release(errorInfo);
        }
        else {
            std::cerr << "Error: Failed to analyze image." << std::endl;
        }
        SIMPLELPR_ReferenceCounted_release(processor);
        SIMPLELPR_ReferenceCounted_release(engine);
        return 1;
    }

    // Get the number of candidates (possible license plates)
    std::cout << "Get the number of candidates (possible license plates)" << std::endl;
    SIMPLELPR_SIZE_T numCandidates = SIMPLELPR_Candidates_numCandidates_get(candidates);
    if (numCandidates == 0) {
        std::cout << "No license plates found." << std::endl;
    }
    else {
        for (SIMPLELPR_SIZE_T i = 0; i < numCandidates; ++i) {
            SIMPLELPR_Handle candidate = SIMPLELPR_Candidates_candidate_get(candidates, i);
            if (candidate) {
                SIMPLELPR_Handle countryMatch = SIMPLELPR_Candidate_countryMatch_get(candidate, 0);
                if (countryMatch) {
                    const wchar_t* plateText = SIMPLELPR_CountryMatch_text_get(countryMatch);
                    std::wcout << L"License Plate: " << plateText << std::endl;
                    SIMPLELPR_ReferenceCounted_release(countryMatch);
                }
                SIMPLELPR_ReferenceCounted_release(candidate);
            }
        }
    }

    // Release resources
    SIMPLELPR_ReferenceCounted_release(candidates);
    SIMPLELPR_ReferenceCounted_release(processor);
    SIMPLELPR_ReferenceCounted_release(engine);

    return 0;
}
