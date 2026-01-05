"""Model registry API endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.auth.firebase import FirebaseUser, get_current_user, get_current_user_optional
from app.models.model import (
    ModelCreate,
    ModelDownloadResponse,
    ModelInfo,
    ModelListResponse,
    ModelType,
    ModelUploadResponse,
    ModelVersion,
)
from app.services.storage import StorageService, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[ModelType] = Query(None, description="Filter by model type"),
    public_only: bool = Query(False, description="Only show public models"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: Optional[FirebaseUser] = Depends(get_current_user_optional),
    storage: StorageService = Depends(get_storage_service),
):
    """
    List available models.

    Returns public models and models owned by the authenticated user.
    Unauthenticated users can only see public models.
    """
    # Get models based on authentication
    if user and not public_only:
        # Get user's models and public models
        user_models = await storage.list_models(owner_id=user.uid)
        public_models = await storage.list_models(public_only=True)

        # Combine and deduplicate
        model_ids = set()
        models = []
        for model in user_models + public_models:
            if model.id not in model_ids:
                model_ids.add(model.id)
                models.append(model)
    else:
        # Only public models
        models = await storage.list_models(public_only=True)

    # Apply type filter
    if model_type:
        models = [m for m in models if m.model_type == model_type]

    # Calculate pagination
    total = len(models)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_models = models[start:end]

    return ModelListResponse(
        models=paginated_models,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_data: ModelCreate,
    user: FirebaseUser = Depends(get_current_user),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Create a new model entry.

    Creates metadata for a new model. Use the upload endpoint
    to add model files/versions.
    """
    model = await storage.create_model(
        owner_id=user.uid,
        name=model_data.name,
        description=model_data.description,
        model_type=model_data.model_type,
        framework=model_data.framework,
        tags=model_data.tags,
        is_public=model_data.is_public,
    )

    logger.info(f"User {user.uid} created model {model.id}")
    return model


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    user: Optional[FirebaseUser] = Depends(get_current_user_optional),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get model details by ID.

    Returns model metadata including version count and download stats.
    """
    model = await storage.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check access permissions
    if not model.is_public:
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for private models",
            )
        if model.owner_id != user.uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this model",
            )

    return model


@router.get("/{model_id}/versions", response_model=List[ModelVersion])
async def list_versions(
    model_id: str,
    user: Optional[FirebaseUser] = Depends(get_current_user_optional),
    storage: StorageService = Depends(get_storage_service),
):
    """
    List all versions of a model.

    Returns version history with checksums and release notes.
    """
    model = await storage.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check access permissions
    if not model.is_public:
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for private models",
            )
        if model.owner_id != user.uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this model",
            )

    return await storage.list_versions(model_id)


@router.post("/{model_id}/upload", response_model=ModelUploadResponse)
async def upload_model(
    model_id: str,
    version: str = Query(..., description="Semantic version (e.g., 1.0.0)"),
    release_notes: Optional[str] = Query(None, description="Release notes for this version"),
    file: UploadFile = File(..., description="Model file (.bin, .pt, .onnx, etc.)"),
    user: FirebaseUser = Depends(get_current_user),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Upload a new model version.

    Uploads a model file and creates version metadata.
    Maximum file size: 500MB (use signed URL for larger files).
    """
    # Verify model exists and user owns it
    model = await storage.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    if model.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to upload to this model",
        )

    # Check if version already exists
    existing_versions = await storage.list_versions(model_id)
    for v in existing_versions:
        if v.version == version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version {version} already exists",
            )

    # Validate file size (500MB limit for direct upload)
    max_size = 500 * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Use signed URL upload for files > 500MB",
        )

    # Reset file position for upload
    await file.seek(0)

    # Upload model
    download_url = await storage.upload_model(
        model_id=model_id,
        version=version,
        file=file,
        user_id=user.uid,
        release_notes=release_notes,
    )

    import hashlib
    checksum = hashlib.sha256(content).hexdigest()

    logger.info(f"User {user.uid} uploaded version {version} of model {model_id}")

    return ModelUploadResponse(
        model_id=model_id,
        version=version,
        download_url=download_url,
        checksum=checksum,
        message="Model uploaded successfully",
    )


@router.get("/{model_id}/{version}/download", response_model=ModelDownloadResponse)
async def get_download_url(
    model_id: str,
    version: str,
    user: Optional[FirebaseUser] = Depends(get_current_user_optional),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get a signed download URL for a model version.

    Returns a time-limited URL for downloading the model file.
    URL expires in 1 hour.
    """
    model = await storage.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check access permissions
    if not model.is_public:
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for private models",
            )
        if model.owner_id != user.uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to download this model",
            )

    try:
        download_url = await storage.get_signed_url(model_id, version)
    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found for model {model_id}",
        )

    # Get version metadata for checksum and size
    versions = await storage.list_versions(model_id)
    version_info = next((v for v in versions if v.version == version), None)

    if not version_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found",
        )

    return ModelDownloadResponse(
        model_id=model_id,
        version=version,
        download_url=download_url,
        expires_in=3600,
        checksum=version_info.checksum,
        file_size_bytes=version_info.file_size_bytes,
    )


@router.post("/{model_id}/upload-url")
async def get_upload_url(
    model_id: str,
    version: str = Query(..., description="Semantic version"),
    user: FirebaseUser = Depends(get_current_user),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get a signed URL for uploading large model files.

    Use this for files larger than 500MB. Upload directly to
    the signed URL using PUT with content-type: application/octet-stream.
    """
    model = await storage.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    if model.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to upload to this model",
        )

    upload_url = await storage.get_upload_signed_url(model_id, version)

    return {
        "upload_url": upload_url,
        "method": "PUT",
        "content_type": "application/octet-stream",
        "expires_in": 3600,
        "instructions": "Upload file with PUT request to the upload_url",
    }
