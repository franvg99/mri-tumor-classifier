"""
Crea el rol de IAM que SageMaker usa para publicar el modelo.

Aplica mínimo privilegio en lugar de la política general
`AmazonSageMakerFullAccess`. El rol puede:

- leer la carpeta `models/` del bucket del proyecto (solo lectura);
- descargar la imagen oficial de PyTorch para inferencia;
- escribir los registros (logs) y las métricas del endpoint.

Solo lo puede usar el servicio SageMaker, y solo desde la propia cuenta
(condición `aws:SourceAccount`).

Alcance: cubre el despliegue. Un Training Job necesitaría además leer los datos
y escribir el modelo en S3.

Uso (desde la raíz del proyecto, con la sesión de `aws login` activa):
    python -m scripts.create_sagemaker_role
"""

import json
import sys

import boto3
import yaml

NOMBRE_ROL = "mri-tumor-sagemaker-role"
NOMBRE_POLITICA = "mri-tumor-despliegue"
CUENTA_IMAGENES_AWS = "763104351884"  # cuenta de AWS que publica los contenedores oficiales


def politica_de_confianza(cuenta):
    """Quién puede usar el rol."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": cuenta}},
            }
        ],
    }


def politica_de_permisos(cuenta, region, bucket):
    """Qué puede hacer el rol."""
    logs = f"arn:aws:logs:{region}:{cuenta}:log-group:/aws/sagemaker/Endpoints/*"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "LeerPaqueteDelModelo",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/models/*",
            },
            {
                # Esta acción no admite restringirse a un recurso.
                "Sid": "AutenticarseParaDescargarImagen",
                "Effect": "Allow",
                "Action": "ecr:GetAuthorizationToken",
                "Resource": "*",
            },
            {
                "Sid": "DescargarImagenDePyTorch",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": f"arn:aws:ecr:{region}:{CUENTA_IMAGENES_AWS}:repository/pytorch-inference",
            },
            {
                "Sid": "EscribirRegistros",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                ],
                "Resource": logs,
            },
            {
                # Tampoco admite recurso: se restringe por espacio de nombres.
                "Sid": "EscribirMetricas",
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {
                    "StringLike": {"cloudwatch:namespace": ["AWS/SageMaker", "/aws/sagemaker/*"]}
                },
            },
        ],
    }


def main() -> int:
    """Crea el rol con su política. Falla si el rol ya existe."""
    with open("config/config.yaml", encoding="utf-8") as f:
        aws = yaml.safe_load(f)["aws"]

    cuenta = boto3.client("sts", region_name=aws["region"]).get_caller_identity()["Account"]
    iam = boto3.client("iam", region_name=aws["region"])

    iam.create_role(
        RoleName=NOMBRE_ROL,
        AssumeRolePolicyDocument=json.dumps(politica_de_confianza(cuenta)),
        Description="SageMaker: despliegue del clasificador de tumores en MRI",
    )
    iam.put_role_policy(
        RoleName=NOMBRE_ROL,
        PolicyName=NOMBRE_POLITICA,
        PolicyDocument=json.dumps(politica_de_permisos(cuenta, aws["region"], aws["s3_bucket"])),
    )
    print(f"Rol creado: {NOMBRE_ROL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
