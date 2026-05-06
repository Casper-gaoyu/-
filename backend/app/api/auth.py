from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import User
from app.services.auth_service import AuthService

auth_service = AuthService()
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials.strip():
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="请先登录")
    return auth_service.get_user_by_token(db, credentials.credentials.strip())
