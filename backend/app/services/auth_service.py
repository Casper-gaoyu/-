import hashlib
import hmac
import re
import secrets
from typing import Final

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.db_models import AuthToken, User

USERNAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{3,50}$")


class AuthService:
    HASH_ITERATIONS = 100_000

    def register(self, db: Session, username: str, password: str) -> dict:
        normalized_username = self._normalize_username(username)
        self._validate_password(password)

        exists = db.execute(select(User).where(User.username == normalized_username)).scalar_one_or_none()
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")

        user = User(
            username=normalized_username,
            password_hash=self._hash_password(password),
        )
        db.add(user)
        db.flush()
        token = self._issue_token(db, user.id)
        db.commit()
        db.refresh(user)
        return {
            "token": token,
            "user": self.serialize_user(user),
        }

    def login(self, db: Session, username: str, password: str) -> dict:
        normalized_username = self._normalize_username(username)
        user = db.execute(select(User).where(User.username == normalized_username)).scalar_one_or_none()
        if not user or not self._verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        token = self._issue_token(db, user.id)
        db.commit()
        return {
            "token": token,
            "user": self.serialize_user(user),
        }

    def logout(self, db: Session, token: str) -> None:
        db.execute(delete(AuthToken).where(AuthToken.token == token))
        db.commit()

    def get_user_by_token(self, db: Session, token: str) -> User:
        auth_token = db.execute(select(AuthToken).where(AuthToken.token == token)).scalar_one_or_none()
        if not auth_token:
            raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")

        user = db.get(User, auth_token.user_id)
        if not user:
            db.execute(delete(AuthToken).where(AuthToken.token == token))
            db.commit()
            raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
        return user

    @staticmethod
    def serialize_user(user: User) -> dict:
        return {
            "id": user.id,
            "username": user.username,
        }

    def _issue_token(self, db: Session, user_id: int) -> str:
        db.execute(delete(AuthToken).where(AuthToken.user_id == user_id))
        token = secrets.token_urlsafe(32)
        db.add(AuthToken(user_id=user_id, token=token))
        return token

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = username.strip()
        if not USERNAME_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="用户名需为 3-50 位，可包含中文、字母、数字、下划线或短横线")
        return normalized

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password.strip()) < 6:
            raise HTTPException(status_code=400, detail="密码长度不能少于 6 位")

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            self.HASH_ITERATIONS,
        ).hex()
        return f"{salt}${digest}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            salt, digest = stored_hash.split("$", 1)
        except ValueError:
            return False

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            self.HASH_ITERATIONS,
        ).hex()
        return hmac.compare_digest(candidate, digest)
