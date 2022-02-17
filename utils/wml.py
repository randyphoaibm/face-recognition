# API_KEY = ""
# SPACE_ID = "23c8a765-48bf-461a-88e9-167dc558227a"
# wml_credentials = {"apikey": API_KEY, "url": "https://us-south.ml.cloud.ibm.com"}
# wml_client = ibm_watson_machine_learning.APIClient(wml_credentials)

# wml_client.set.default_project("000a8ff7-39d8-45d5-beda-46aae640d87f")
# wml_client.data_assets.list()
# wml_client.data_assets.download("f1633884-4408-4235-97e3-fc7debe0ca9a", "1.jpg")
# wml_client.data_assets.create(name="dlib_lib", file_path="dlib.so")

# wml_client.spaces.list(limit=10)
# wml_client.set.default_space(SPACE_ID)

import ibm_watson_machine_learning

def deploy_model(wml_client, model, model_name, model_deployment_name):
    for x in wml_client.deployments.get_details()["resources"]:
        if x["metadata"]["name"] == model_deployment_name:
            wml_client.deployments.delete(x["metadata"]["id"])

    for x in wml_client.repository.get_model_details()["resources"]:
        if x["metadata"]["name"] == model_name:
            wml_client.repository.delete(x["metadata"]["id"])

    meta_props = {
        wml_client.repository.ModelMetaNames.NAME: model_name,
        wml_client.repository.ModelMetaNames.TYPE: "tensorflow_2.4",
        wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: wml_client.software_specifications.get_id_by_name("tensorflow_2.4-py3.8"),
        wml_client.repository.ModelMetaNames.TF_MODEL_PARAMS: {"save_format": "h5"},
    }

    model_details = wml_client.repository.store_model(model=model, meta_props=meta_props)
    model_uid = wml_client.repository.get_model_uid(model_details)

    meta_props = {wml_client.deployments.ConfigurationMetaNames.NAME: model_deployment_name, wml_client.deployments.ConfigurationMetaNames.ONLINE: {}}
    deployment_details = wml_client.deployments.create(model_uid, meta_props=meta_props)
    deployment_uid = wml_client.deployments.get_uid(deployment_details)

    return deployment_uid


def deploy_function(wml_client, function, function_name, function_deployment_name, software_spec_uid):
    for x in wml_client.deployments.get_details()["resources"]:
        if x["metadata"]["name"] == function_deployment_name:
            wml_client.deployments.delete(x["metadata"]["id"])

    for x in wml_client.repository.get_function_details()["resources"]:
        if x["metadata"]["name"] == function_name:
            wml_client.repository.delete(x["metadata"]["id"])

    meta_props = {
        wml_client.repository.FunctionMetaNames.NAME: function_name,
        wml_client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    }
    function_details = wml_client.repository.store_function(meta_props=meta_props, function=function)

    meta_props = {
        wml_client.deployments.ConfigurationMetaNames.NAME: function_deployment_name,
        wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    function_uid = wml_client.repository.get_function_uid(function_details)
    deployment_details = wml_client.deployments.create(function_uid, meta_props=meta_props)
    deployment_uid = wml_client.deployments.get_uid(deployment_details)

    return deployment_uid


def create_software_spec(wml_client, software_spec_name, conda_yaml):
    software_spec_uid = wml_client.software_specifications.get_uid_by_name(software_spec_name)
    if software_spec_uid != "Not Found":
        software_spec_details = wml_client.software_specifications.get_details(software_spec_uid)
        package_ext_uid = software_spec_details["entity"]["software_specification"]["package_extensions"][0]["metadata"]["asset_id"]
        wml_client.package_extensions.delete(package_ext_uid)
        wml_client.software_specifications.delete(software_spec_uid)

    meta_props = {wml_client.package_extensions.ConfigurationMetaNames.NAME: "dlib env", wml_client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"}
    package_ext_details = wml_client.package_extensions.store(meta_props=meta_props, file_path=conda_yaml)
    package_ext_uid = wml_client.package_extensions.get_uid(package_ext_details)

    meta_props = {
        wml_client.software_specifications.ConfigurationMetaNames.NAME: software_spec_name,
        wml_client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": wml_client.software_specifications.get_uid_by_name("default_py3.8")},
    }
    software_spec_details = wml_client.software_specifications.store(meta_props=meta_props)
    software_spec_uid = wml_client.software_specifications.get_uid(software_spec_details)
    wml_client.software_specifications.add_package_extension(software_spec_uid, package_ext_uid)

    software_spec_details = wml_client.software_specifications.get_details(software_spec_uid)
    software_spec_uid = wml_client.software_specifications.get_uid(software_spec_details)
    return software_spec_uid


def upload_lib(wml_client, name, file_path):
    for x in wml_client.data_assets.get_details()["resources"]:
        if x["metadata"]["name"] == name:
            wml_client.data_assets.delete(x["metadata"]["asset_id"])
    asset_details = wml_client.data_assets.create(name=name, file_path=file_path)
    asset_uid = wml_client.data_assets.get_uid(asset_details)
    return asset_uid


def get_asset_uid(wml_client, asset_name):
    for x in wml_client.data_assets.get_details()["resources"]:
        if x["metadata"]["name"] == asset_name:
            return x["metadata"]["asset_id"]


def deploy_dlib(wml_credentials, SPACE_ID, conda_yaml, lib_path):
    wml_client = ibm_watson_machine_learning.APIClient(wml_credentials)
    wml_client.set.default_space(SPACE_ID)

    SOFTWARE_SPEC_NAME = "dlib"
    software_spec_uid = create_software_spec(wml_client=wml_client, software_spec_name=SOFTWARE_SPEC_NAME, conda_yaml=conda_yaml)
    lib_uid = upload_lib(wml_client=wml_client, name="dlib_lib", file_path=lib_path)
    software_spec_uid

    FUNCTION_NAME = "dlib Function"
    FUNCTION_DEPLOYMENT_NAME = "dlib Function Deployment"

    params = {
        "wml_credentials": wml_credentials,
        "lib_uid": lib_uid,
        "SPACE_ID": SPACE_ID,
    }

    def func(params=params):
        import numpy as np
        import ibm_watson_machine_learning

        wml_client = ibm_watson_machine_learning.APIClient(params["wml_credentials"])
        wml_client.set.default_space(params["SPACE_ID"])
        wml_client.data_assets.download(params["lib_uid"], "_dlib_pybind11.cpython-38-x86_64-linux-gnu.so")
        import _dlib_pybind11 as dlib

        detector = dlib.get_frontal_face_detector()

        def score(payload):
            img = payload["input_data"][0]["values"]
            detections = detector(np.array(img, dtype=np.uint8))
            detections = [(x.left(), x.top(), x.right(), x.bottom()) for x in detections]
            return {"predictions": [{"values": detections,}]}

        return score

    function_deployment_uid = deploy_function(
        wml_client=wml_client, function=func, function_name=FUNCTION_NAME, function_deployment_name=FUNCTION_DEPLOYMENT_NAME, software_spec_uid=software_spec_uid
    )

    return function_deployment_uid
