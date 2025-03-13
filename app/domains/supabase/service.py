import os
import tempfile
from supabase.client import Client
from litestar.datastructures import UploadFile
from configs.supabase import provide_supabase_client
from litestar.exceptions import InternalServerException
from storage3.utils import StorageException

class SupabaseService:
    supabase_client: Client

    def __init__(self):
        self.supabase_client = provide_supabase_client()

    async def upload_file(
        self, file: UploadFile, bucket_name: str, name: str | None
    ) -> str:
        # Read the image file content
        image_content = await file.read()
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        # If a name is provided, use it; otherwise, use file.filename (with extension)
        name = f"{name}.{file_extension}" if name else file.filename

        fd, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(image_content)
            ff.close()
        os.close(fd)
        try:
            # Upload the file to Supabase Storage using the closed temporary file
            upload_result = self.supabase_client.storage.from_(bucket_name).upload(
                name,
                fname,
                {"content-type": file.content_type},
            )
        except StorageException as exc:
            print(exc)
            # Get the public URL of the uploaded file
        finally:
            pass
        public_url = self.supabase_client.storage.from_(bucket_name).get_public_url(name)
        return public_url


async def provide_supabase_service() -> SupabaseService:
    return SupabaseService()
