"""Tests unitarios para scripts/deploy_endpoint.py y src/inference/predictor.py.

No se conectan a AWS: un cliente falso registra los pedidos.
"""

import io
import json

import jmespath
import pytest
from botocore.exceptions import WaiterError

from scripts import deploy_endpoint
from src.inference import predictor

LISTADOS = {
    "list_endpoints": ("endpoint", "Endpoints", "EndpointName"),
    "list_endpoint_configs": ("configuracion", "EndpointConfigs", "EndpointConfigName"),
    "list_models": ("modelo", "Models", "ModelName"),
}


class SageMakerFalso:
    """Simula la creación y el borrado de recursos, y anota el orden de los pedidos."""

    def __init__(self, existentes=(), falla_al_iniciar=False):
        self.existentes = set(existentes)
        self.falla_al_iniciar = falla_al_iniciar
        self.pedidos = []

    def get_paginator(self, operacion):
        recurso, lista, campo = LISTADOS[operacion]
        # Incluye un nombre parecido: la búsqueda de AWS es por coincidencia parcial.
        nombres = [f"{deploy_endpoint.NOMBRE}-otro"]
        if recurso in self.existentes:
            nombres.append(deploy_endpoint.NOMBRE)
        pagina = {lista: [{campo: n} for n in nombres]}

        class Paginador:
            def paginate(self, **_):
                return self

            def search(self, expresion):
                return iter(jmespath.search(expresion, pagina))

        return Paginador()

    def get_waiter(self, nombre):
        falla = self.falla_al_iniciar and nombre == "endpoint_in_service"

        class Espera:
            def wait(self, **_):
                if falla:
                    raise WaiterError(name=nombre, reason="Failed", last_response={})

        return Espera()

    def _anotar(self, pedido, recurso, crea):
        self.pedidos.append(pedido)
        (self.existentes.add if crea else self.existentes.discard)(recurso)

    def create_model(self, **_):
        self._anotar("crear modelo", "modelo", True)

    def create_endpoint_config(self, **_):
        self._anotar("crear configuracion", "configuracion", True)

    def create_endpoint(self, **_):
        self._anotar("crear endpoint", "endpoint", True)

    def delete_model(self, **_):
        self._anotar("borrar modelo", "modelo", False)

    def delete_endpoint_config(self, **_):
        self._anotar("borrar configuracion", "configuracion", False)

    def delete_endpoint(self, **_):
        self._anotar("borrar endpoint", "endpoint", False)


def crear(sm):
    deploy_endpoint.crear(sm, "arn:rol", "imagen", "s3://bucket/model.tar.gz", "ml.m5.xlarge")


def test_crear_sigue_el_orden_que_exige_sagemaker():
    sm = SageMakerFalso()
    crear(sm)
    assert sm.pedidos == ["crear modelo", "crear configuracion", "crear endpoint"]


def test_crear_elimina_todo_si_el_endpoint_no_arranca():
    """Un endpoint fallido no puede quedar facturando."""
    sm = SageMakerFalso(falla_al_iniciar=True)
    with pytest.raises(WaiterError):
        crear(sm)
    assert sm.existentes == set()


def test_crear_no_duplica_recursos():
    sm = SageMakerFalso(existentes={"endpoint"})
    with pytest.raises(RuntimeError, match="borrar"):
        crear(sm)
    assert sm.pedidos == []


def test_borrar_empieza_por_el_endpoint():
    sm = SageMakerFalso(existentes={"endpoint", "configuracion", "modelo"})
    assert deploy_endpoint.borrar(sm) == ["endpoint", "configuracion", "modelo"]
    assert sm.pedidos[0] == "borrar endpoint"
    assert sm.existentes == set()


def test_borrar_sin_recursos_no_falla():
    assert deploy_endpoint.borrar(SageMakerFalso()) == []


def test_predict_image_envia_el_formato_correcto(monkeypatch, tmp_path):
    enviado = {}

    class RuntimeFalso:
        def invoke_endpoint(self, **pedido):
            enviado.update(pedido)
            return {"Body": io.BytesIO(json.dumps({"clase": "glioma"}).encode())}

    monkeypatch.setattr(predictor.boto3, "client", lambda *a, **k: RuntimeFalso())
    imagen = tmp_path / "corte.JPG"
    imagen.write_bytes(b"contenido")

    assert predictor.predict_image(str(imagen), "mi-endpoint") == {"clase": "glioma"}
    assert enviado["ContentType"] == "image/jpeg"
    assert enviado["Body"] == b"contenido"


def test_predict_image_rechaza_otras_extensiones(tmp_path):
    with pytest.raises(ValueError, match="Extensión no admitida"):
        predictor.predict_image(str(tmp_path / "corte.dcm"), "mi-endpoint")
