from pydantic import BaseModel


class RequestText(BaseModel):
    text: str