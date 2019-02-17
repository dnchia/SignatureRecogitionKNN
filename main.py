from signature_verification_service import SignatureVerificationService


MAX_CLASSES = 58

def main():
    signature_verification_service = SignatureVerificationService(
        signatures_directory="signatures",
        num_of_classes=MAX_CLASSES,
        signatures_per_class=20)

    signature_verification_service.build_model_with_signatures_features(
        division_between_training_and_test=0.8,
        number_of_executions=100
    )
    return


if __name__ == '__main__':
    main()
    exit(0)
