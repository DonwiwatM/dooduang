from fastapi import APIRouter, Depends, HTTPException


router = APIRouter(
    prefix="/feature_extract",
    tags=["feature_extract"],
    responses={404: {"description": "Not found"}},
)

fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


@router.get("/")
async def read_items():
    return fake_items_db

