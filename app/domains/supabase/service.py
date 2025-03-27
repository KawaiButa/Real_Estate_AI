import os
import tempfile
import uuid
from supabase.client import Client
from litestar.datastructures import UploadFile
from configs.supabase import provide_supabase_client
from database.models.image import Image
from litestar.exceptions import InternalServerException
from storage3.utils import StorageException


class SupabaseService:
    supabase_client: Client
    bucket_name: str

    def __init__(self, bucket_name: str = ""):
        self.supabase_client = provide_supabase_client()
        self.bucket_name = bucket_name

    async def upload_file(
        self, file: UploadFile, bucket_name: str | None = None, name: str | None = None
    ) -> str:
        if not bucket_name:
            bucket_name = self.bucket_name
        image_content = await file.read()
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        name = f"{name}.{file_extension}" if name else str(uuid.uuid4())

        fd, fname = tempfile.mkstemp()
        with open(fname, "wb") as ff:
            ff.write(image_content)
            ff.close()
        os.close(fd)
        try:
            upload_result = self.supabase_client.storage.from_(bucket_name).upload(
                name,
                fname,
                {"content-type": file.content_type},
            )
        except StorageException as exc:
            print(exc)
        finally:
            pass
        public_url = self.supabase_client.storage.from_(bucket_name).get_public_url(
            name
        )
        return public_url

    async def delete_image(self, image: Image, bucket_name: str | None = None) -> None:
        if not bucket_name:
            bucket_name = self.bucket_name
        try:
            bucket_name = bucket_name
            filename = image.url.split("/")[-1]
            self.supabase_client.storage.from_(bucket_name).remove([filename])
        except Exception as e:
            raise InternalServerException(f"Failed to delete image from storage: {e}")

    async def update_image(
        self, image: Image, file: UploadFile, bucket_name: str | None = None
    ) -> str:
        if not bucket_name:
            bucket_name = self.bucket_name
        try:
            old_filename = image.url.split("/")[-1]
            self.supabase_client.storage.from_(bucket_name).remove([old_filename])
        except Exception as e:
            raise InternalServerException(
                f"Failed to delete old image from storage: {e}"
            )
        # Prepare the new file for upload.
        new_image_content = await file.read()
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        # Reuse the old filename if available; otherwise, generate a new one.
        filename = old_filename if old_filename else f"{uuid.uuid4()}.{file_extension}"
        # Write file content to a temporary file.
        fd, tmp_path = tempfile.mkstemp()
        try:
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(new_image_content)
        finally:
            os.close(fd)
        # Upload the new file to Supabase Storage.
        try:
            self.supabase_client.storage.from_(bucket_name).upload(
                filename,
                tmp_path,
                {"content-type": file.content_type},
            )
        except StorageException as exc:
            raise InternalServerException(f"Failed to upload new image: {exc}")
        finally:
            os.remove(tmp_path)

        public_url = self.supabase_client.storage.from_(bucket_name).get_public_url(
            filename
        )
        return public_url


def provide_supabase_service(bucket_name: str) -> SupabaseService:
    return SupabaseService(bucket_name=bucket_name)
