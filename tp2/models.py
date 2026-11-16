from pydantic import BaseModel


class Prompt(BaseModel):
    message: str


class Translation(BaseModel):
    translation: str
