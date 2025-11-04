from mysql.connector import Error, pooling
import configparser

config = configparser.ConfigParser()
config.read('server/setting.conf')

db0 = {
    "host": config['database']['host'],
    "port": config['database']['port'],
    "user": config['database']['user'],
    "password": config['database']['password'],
    "database": config['database']['database'],
    "autocommit": config.getboolean('database','autocommit') #must be converted
}

def init_db_pool():
    global pool
    try:
        pool = pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=25,
            **db0
        )
        return pool
    except Error as e:
        print(f"[ERROR] Database pool creation failed: {e}")
        return None


pool = init_db_pool()
