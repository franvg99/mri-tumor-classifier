"""
Publica el modelo entrenado como SageMaker Real-Time Endpoint, consulta su estado
y lo elimina.

IMPORTANTE: un endpoint activo factura por hora aunque no reciba tráfico y no se
apaga solo (ml.m5.xlarge: US$ 0,23 por hora en us-east-2, consultado el
25/11/2026). Hay que ejecutar `borrar` apenas se termina de probarlo.

Publicar requiere crear tres recursos, en este orden:

- modelo: el paquete de S3 más el contenedor de PyTorch que lo ejecuta;
- configuración: en qué tipo de máquina y cuántas;
- endpoint: la máquina encendida. Es el único de los tres que factura.

Uso (desde la raíz del proyecto, con la sesión de `aws login` activa):
    python -m scripts.deploy_endpoint crear
    python -m scripts.deploy_endpoint estado
    python -m scripts.deploy_endpoint borrar
"""

import argparse
import sys

import boto3
import yaml
from botocore.exceptions import ClientError, WaiterError

from scripts.create_sagemaker_role import CUENTA_IMAGENES_AWS, NOMBRE_ROL

NOMBRE = "mri-tumor-classifier"  # mismo nombre para el modelo, la configuración y el endpoint
IMAGEN = "pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker"


def leer_config():
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recursos_existentes(sm):
    """Indica cuáles de los tres recursos existen.

    Returns:
        Diccionario `{"endpoint": bool, "configuracion": bool, "modelo": bool}`.
    """
    consultas = {
        "endpoint": ("list_endpoints", "Endpoints[].EndpointName"),
        "configuracion": ("list_endpoint_configs", "EndpointConfigs[].EndpointConfigName"),
        "modelo": ("list_models", "Models[].ModelName"),
    }
    return {
        recurso: NOMBRE in sm.get_paginator(operacion).paginate(NameContains=NOMBRE).search(campo)
        for recurso, (operacion, campo) in consultas.items()
    }


def borrar(sm):
    """Elimina los recursos que existan, primero el endpoint (el que factura).

    Returns:
        La lista de recursos eliminados.
    """
    existentes = recursos_existentes(sm)
    if existentes["endpoint"]:
        sm.delete_endpoint(EndpointName=NOMBRE)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=NOMBRE)
    if existentes["configuracion"]:
        sm.delete_endpoint_config(EndpointConfigName=NOMBRE)
    if existentes["modelo"]:
        sm.delete_model(ModelName=NOMBRE)
    return [recurso for recurso, existe in existentes.items() if existe]


def crear(sm, rol_arn, imagen, paquete_s3, tipo_maquina):
    """Publica el modelo y espera a que esté listo para responder.

    Si la publicación falla, elimina lo creado antes de informar el error: un
    endpoint a medio crear también puede facturar.
    """
    if any(recursos_existentes(sm).values()):
        raise RuntimeError("Ya existen recursos con este nombre: ejecutar `borrar` primero")

    try:
        sm.create_model(
            ModelName=NOMBRE,
            ExecutionRoleArn=rol_arn,
            PrimaryContainer={
                "Image": imagen,
                "ModelDataUrl": paquete_s3,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
            },
        )
        sm.create_endpoint_config(
            EndpointConfigName=NOMBRE,
            ProductionVariants=[
                {
                    "VariantName": "principal",
                    "ModelName": NOMBRE,
                    "InstanceType": tipo_maquina,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        sm.create_endpoint(EndpointName=NOMBRE, EndpointConfigName=NOMBRE)
        sm.get_waiter("endpoint_in_service").wait(EndpointName=NOMBRE)
    except (ClientError, WaiterError):
        borrar(sm)
        raise


def main(argv=None) -> int:
    """Punto de entrada: ejecuta la acción pedida e informa el resultado."""
    parser = argparse.ArgumentParser(description="Despliegue del modelo en SageMaker")
    parser.add_argument("accion", choices=["crear", "estado", "borrar"])
    accion = parser.parse_args(argv).accion

    config = leer_config()
    aws = config["aws"]
    sm = boto3.client("sagemaker", region_name=aws["region"])

    try:
        if accion == "crear":
            rol = boto3.client("iam", region_name=aws["region"]).get_role(RoleName=NOMBRE_ROL)
            print("Publicando el modelo (suele tardar varios minutos)...", flush=True)
            crear(
                sm,
                rol_arn=rol["Role"]["Arn"],
                imagen=f"{CUENTA_IMAGENES_AWS}.dkr.ecr.{aws['region']}.amazonaws.com/{IMAGEN}",
                paquete_s3=f"s3://{aws['s3_bucket']}/{aws['s3_models_prefix']}model.tar.gz",
                tipo_maquina=config["inference"]["endpoint_instance_type"],
            )
            print(f"Endpoint activo: {NOMBRE}. Factura por hora: ejecutar `borrar` al terminar.")
        elif accion == "borrar":
            print(f"Eliminados: {borrar(sm) or 'nada, no había recursos'}")

        existentes = recursos_existentes(sm)
        print(f"Recursos: {existentes}")
        if existentes["endpoint"]:
            print("Estado:", sm.describe_endpoint(EndpointName=NOMBRE)["EndpointStatus"])
        return 0
    except (ClientError, WaiterError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrumpido. El endpoint puede seguir activo: ejecutar `borrar`.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
