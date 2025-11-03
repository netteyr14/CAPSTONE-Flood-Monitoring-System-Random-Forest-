from mysql.connector import Error, pooling

db0 = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Ambin123456123456", # change this to your own mysql root password
    "database": "random_forest",
    "autocommit": True
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
