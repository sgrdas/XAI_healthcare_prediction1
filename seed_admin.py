"""Create an administrator account using credentials entered interactively."""

from getpass import getpass

from app import create_app, db
from models.admin import Admin


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        full_name = input("Administrator name: ").strip()
        email = input("Administrator email: ").strip().lower()
        if not full_name or not email:
            raise SystemExit("Administrator name and email are required.")
        if Admin.query.filter_by(email=email).first():
            raise SystemExit("An administrator with that email already exists.")

        password = getpass("Administrator password (minimum 12 characters): ")
        if len(password) < 12:
            raise SystemExit("Administrator password must be at least 12 characters.")
        if len(password.encode("utf-8")) > 72:
            raise SystemExit("Administrator password must be at most 72 UTF-8 bytes.")
        if password != getpass("Confirm administrator password: "):
            raise SystemExit("Passwords do not match.")

        admin = Admin(full_name=full_name, email=email)
        admin.password = password
        db.session.add(admin)
        db.session.commit()
        print(f"Administrator account created for {email}.")


if __name__ == "__main__":
    main()
