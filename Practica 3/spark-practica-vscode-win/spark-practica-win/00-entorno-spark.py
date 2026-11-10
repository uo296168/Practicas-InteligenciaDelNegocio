# -----------------------------------------------------------------------------
# 00-entorno-spark.py
# Se ejecuta automaticamente al arrancar el kernel de Python (IPython) y deja
# creadas, como en un notebook de Databricks:
#   spark      -> la SparkSession
#   sc         -> el SparkContext
#   display()  -> muestra un DataFrame de Spark como tabla
# Si delta-spark esta instalado, la sesion admite tablas Delta.
# No cree otra SparkSession en el notebook: use la variable spark.
# -----------------------------------------------------------------------------
import os
import sys
import shutil

# 1) Spark debe usar el mismo Python que el notebook (imprescindible en
#    Windows y en entornos conda/venv).
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# 2) Localizar Java si JAVA_HOME no esta definido.
if not os.environ.get("JAVA_HOME"):
    _exe = "java.exe" if os.name == "nt" else "java"
    for _c in (
        os.path.join(sys.prefix, "bin", _exe),             # conda (Linux/Mac)
        os.path.join(sys.prefix, "Library", "bin", _exe),  # conda (Windows)
        shutil.which("java"),                              # Java del sistema
    ):
        if _c and os.path.exists(_c):
            os.environ["JAVA_HOME"] = os.path.dirname(
                os.path.dirname(os.path.realpath(_c)))
            break

# 3) Solo en Windows: usar winutils.exe/hadoop.dll si estan en la carpeta
#    "hadoop\bin" junto al entorno virtual (C:\spark-practica\hadoop\bin).
#    Son necesarios para escribir ficheros (df.write, tablas Delta).
if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
    _hadoop = os.path.join(os.path.dirname(sys.prefix), "hadoop")
    if os.path.exists(os.path.join(_hadoop, "bin", "winutils.exe")):
        os.environ["HADOOP_HOME"] = _hadoop
        os.environ["PATH"] = (os.path.join(_hadoop, "bin") + os.pathsep
                              + os.environ.get("PATH", ""))

try:
    from pyspark.sql import SparkSession, DataFrame as _SparkDataFrame
except ImportError:
    pass  # Kernel sin PySpark: no se hace nada.
else:
    # Memoria del driver: 2g por defecto; se cambia con la variable de
    # entorno SPARK_DRIVER_MEMORY (por ejemplo, 1g en SageMaker).
    _mem = os.environ.get("SPARK_DRIVER_MEMORY", "2g")
    try:
        _builder = (
            SparkSession.builder
            .appName("practica-spark")
            .master("local[*]")
            .config("spark.driver.memory", _mem)
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
        )
        # 4) Delta Lake (si el paquete delta-spark esta instalado).
        try:
            import glob as _glob
            import pyspark as _pyspark
            from delta import configure_spark_with_delta_pip as _con_delta
        except ImportError:
            pass
        else:
            _builder = (
                _builder
                .config("spark.sql.extensions",
                        "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog",
                        "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            )
            # En la imagen Docker los .jar de Delta ya vienen incluidos; en
            # SageMaker y Windows se descargan de Internet la primera vez.
            _jars = os.path.join(os.path.dirname(_pyspark.__file__), "jars")
            if not _glob.glob(os.path.join(_jars, "*delta-spark*.jar")):
                _builder = _con_delta(_builder)
        spark = _builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        sc = spark.sparkContext
    except Exception as _e:
        print("*** No se ha podido iniciar Spark. Compruebe que Java esta "
              "instalado y, si usa Delta por primera vez, que hay conexion a "
              "Internet. Detalle:", _e)
    else:
        from IPython.display import display as _ipy_display

        def display(obj, limit=1000):
            """Equivalente sencillo al display() de Databricks."""
            if isinstance(obj, _SparkDataFrame):
                _ipy_display(obj.limit(limit).toPandas())
            else:
                _ipy_display(obj)

        print(f"Spark {spark.version} listo (memoria del driver: {_mem}). "
              "Disponibles: spark, sc, display()")
