from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configurar la URL de conexión a la base de datos MySQL (usando el puerto 3307)
SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://user2:password2@localhost:3307/database2"

# Crear el motor de la base de datos
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Crear la sesión de SQLAlchemy para interactuar con la base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crear la base de datos declarativa
Base = declarative_base()
