try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
except ImportError:  # Permite ejecutar el proyecto aunque PyMongo no esté instalado aún.
    MongoClient = None
    PyMongoError = Exception
    ServerSelectionTimeoutError = Exception

from src.config import MONGO_URI, DB_NAME, COLL_INPUT, COLL_RESULTS, COLL_CONFIG


def get_database():
    """
    Devuelve la base de datos de MongoDB.

    Si PyMongo no está instalado o MongoDB no está iniciado, devuelve None para
    que el pipeline pueda continuar generando CSV, modelo, métricas y gráficos.
    """
    if MongoClient is None:
        print(
            "Aviso: PyMongo no está instalado. "
            "Se omite la escritura en base de datos."
        )
        return None

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return client[DB_NAME]
    except ServerSelectionTimeoutError:
        print(
            "Aviso: MongoDB no está disponible. "
            "Se omite la escritura en base de datos."
        )
        return None


def get_collections():
    db = get_database()
    if db is None:
        return None

    return {
        "datos_entrada": db[COLL_INPUT],
        "resultados_modelo": db[COLL_RESULTS],
        "configuracion_modelo": db[COLL_CONFIG],
    }


def insert_processed_data(records):
    collections = get_collections()
    if collections is None or not records:
        return

    try:
        collections["datos_entrada"].insert_many(records)
    except PyMongoError as exc:
        print(f"Aviso: no se pudieron insertar los datos procesados en MongoDB: {exc}")


def insert_model_results(result_doc):
    collections = get_collections()
    if collections is None:
        return

    try:
        collections["resultados_modelo"].insert_one(result_doc)
    except PyMongoError as exc:
        print(f"Aviso: no se pudieron insertar los resultados en MongoDB: {exc}")


def insert_model_config(config_doc):
    collections = get_collections()
    if collections is None:
        return

    try:
        collections["configuracion_modelo"].insert_one(config_doc)
    except PyMongoError as exc:
        print(f"Aviso: no se pudo insertar la configuración del modelo en MongoDB: {exc}")
