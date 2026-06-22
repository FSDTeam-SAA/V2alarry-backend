from app.models.user import User
from app.utils.security import has_password, verify_password
from app.utils.jwt import create_access_token

class AuthService: 

    def register_user(self, data):
        user = User(
            email=data.email,
            full_name=data.full_name,
            hashed_password=has_password(data.password),
        )
        return user
    
    def authenticate_user(self, user, password:str):
        if not user: 
            return False
        
        if not verify_password(password, user.hashed_password):
            return False
        
        return user
    
    def create_token(self, user):
        return create_access_token(
            {
                "sub": str(user.id), "email": user.email
            }
        )