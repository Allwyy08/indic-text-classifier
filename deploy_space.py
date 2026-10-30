"""
Hugging Face Space Deployment Tool for IndicBERT Tamil Classifier.

SAFETY RULES ENFORCED:
- Refuses to deploy to the old Space ('Allwynnoble7/tamil-indicbert-classifier').
- Requires explicit user input of the NEW Hugging Face username.
- Verifies authentication credentials belong to the new account.
- Never hardcodes access tokens.
- Supports dry-run validation.

Usage:
  # Dry-run validation of Space configuration and files:
  python deploy_space.py --dry-run

  # Deploy to new Space once new username is confirmed:
  python deploy_space.py --username YOUR_NEW_USERNAME --token YOUR_NEW_TOKEN
"""

import os
import sys
import yaml
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FORBIDDEN_SPACES = [
    "Allwynnoble7/tamil-indicbert-classifier",
    "allwynnoble7/tamil-indicbert-classifier"
]
FORBIDDEN_USERS = ["Allwynnoble7", "allwynnoble7"]


def validate_environment_for_deployment(target_username: str) -> bool:
    """Verifies that the target Space and username conform to the safety rules."""
    if target_username in FORBIDDEN_USERS:
        logger.error(f"VIOLATION: Deployment to old user '{target_username}' is STRICTLY FORBIDDEN.")
        return False

    target_space_id = f"{target_username}/indicbert-tamil-classifier"
    if target_space_id in FORBIDDEN_SPACES:
        logger.error(f"VIOLATION: Deployment to '{target_space_id}' is STRICTLY FORBIDDEN.")
        return False

    return True


def verify_files_present():
    """Ensures all necessary files for Hugging Face Streamlit Space exist."""
    required_files = [
        "app.py",
        "requirements.txt",
        "configs/config.yaml",
        "src/__init__.py",
        "src/data_utils.py",
        "src/model_utils.py",
        "src/inference.py"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        logger.error(f"Missing required files for Space: {missing}")
        return False
    logger.info("All required application files are present for Space deployment.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy IndicBERT Tamil Classifier to Hugging Face Space")
    parser.add_argument("--username", type=str, default=None, help="Your NEW Hugging Face username")
    parser.add_argument("--token", type=str, default=None, help="Your NEW Hugging Face write token (or set HF_TOKEN env var)")
    parser.add_argument("--space-name", type=str, default="indicbert-tamil-classifier", help="Target Space name")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without uploading")
    args = parser.parse_args()

    print("=" * 70)
    print("HUGGING FACE SPACE DEPLOYMENT VERIFICATION")
    print("=" * 70)

    files_ok = verify_files_present()
    if not files_ok:
        sys.exit(1)

    if args.dry_run:
        print("\n[SUCCESS] Dry-run complete! Application structure is valid and ready for Space deployment.")
        print("To deploy when your new account is ready, execute:")
        print(f"  python deploy_space.py --username <NEW_USERNAME> --token <NEW_WRITE_TOKEN>")
        return

    if not args.username:
        logger.error("Please provide --username with your NEW Hugging Face account name.")
        sys.exit(1)

    if not validate_environment_for_deployment(args.username):
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("No access token provided. Pass --token or set the HF_TOKEN environment variable.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, whoami
        api = HfApi(token=token)
        user_info = whoami(token=token)
        actual_user = user_info.get("name")

        if actual_user in FORBIDDEN_USERS:
            logger.error(f"CRITICAL: The provided token belongs to '{actual_user}', which is forbidden!")
            sys.exit(1)

        repo_id = f"{args.username}/{args.space_name}"
        logger.info(f"Authenticated as: {actual_user}")
        logger.info(f"Target Space repository: {repo_id}")

        # Create space if not exists
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True
        )
        logger.info(f"Target space '{repo_id}' ready. Uploading workspace files...")

        # Upload files
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["data/raw/*", ".pytest_cache/*", "__pycache__/*", "*.git/*", ".venv/*"]
        )

        logger.info(f"Deployment complete! View your Space at: https://huggingface.co/spaces/{repo_id}")

    except ImportError:
        logger.error("huggingface_hub package is required for remote deployment.")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")


if __name__ == "__main__":
    main()

