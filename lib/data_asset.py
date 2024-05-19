from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Datastore
from azure.core.exceptions import ResourceNotFoundError


def get_or_create_data_asset(client: MLClient, path: str, asset_type: str, name: str, version: str,
                             description: str) -> Data:
    """
    :param client: MLClient object that is used to interact with the ML platform.
    :param path: The path of the data asset.
    :param asset_type: The type of the data asset.
    :param name: The name of the data asset.
    :param version: The version of the data asset.
    :param description: The description of the data asset.
    :return: The Data object representing the data asset.

    This method takes in the necessary parameters to create or retrieve a data asset from the ML platform.
    It first creates a Data object using the provided parameters.
    Then, it tries to retrieve the data asset using the provided name and version. If the data asset is found, it is returned.
    If not found, it creates the data asset and returns it.

    Example usage:
    ```
    client = MLClient()
    path = "/data/samples"
    asset_type = "csv"
    name = "sample_data"
    version = "1.0"
    description = "Sample data for testing"

    data_asset = get_or_create_data_asset(client, path, asset_type, name, version, description)
    ```
    """
    data_asset = Data(
        path=path,
        type=asset_type,
        description=description,
        name=name,
        version=version,
    )
    try:
        ds = client.data.get(data_asset.name, data_asset.version)
        print(f'datastore name: {ds.name} version: {ds.version} get it')
        return ds
    except ResourceNotFoundError:
        print(f'datastore name: {data_asset.name} version: {data_asset.version} not exist create and get it')
        client.data.create_or_update(data_asset)
        return client.data.get(data_asset.name, data_asset.version)


def get_or_create_table_data_asset(client: MLClient, mltable_folder: str, name: str, version: str,
                                   description: str) -> Data:
    """
    Get or create a table data asset.

    :param client: The MLClient instance used for communication with the ML service.
    :param mltable_folder: The path to the folder where the table data asset is located.
    :param name: The name of the table data asset.
    :param version: The version of the table data asset.
    :param description: The description of the table data asset.
    :return: The table data asset object.
    """
    return get_or_create_data_asset(client, mltable_folder, AssetTypes.MLTABLE, name, version, description)


def get_or_create_folder_data_asset(client: MLClient, folder_path: str, name: str, version: str,
                                    description: str) -> Data:
    """
    Retrieves or creates a folder data asset.

    :param client: The MLClient object used to interact with the machine learning service.
    :param folder_path: The path of the folder containing the data asset.
    :param name: The name of the data asset.
    :param version: The version of the data asset.
    :param description: The description of the data asset.
    :return: The retrieved or created folder data asset.
    """
    return get_or_create_data_asset(client, folder_path, AssetTypes.URI_FOLDER, name, version, description)


def get_or_create_file_data_asset(client: MLClient, file_path: str, name: str, version: str, description: str) -> Data:
    """
    Get or create a file data asset.

    :param client: The MLClient instance used for interfacing with the machine learning platform.
    :param file_path: The path of the file.
    :param name: The name of the data asset.
    :param version: The version of the data asset.
    :param description: The description of the data asset.

    :return: The Data object representing the file data asset.
    """
    return get_or_create_data_asset(client, file_path, AssetTypes.URI_FILE, name, version, description)


def build_azure_data_path(client: MLClient, datastore: Datastore, storage_path: Optional[str] = None) -> str:
    """
    Build the Azure data path based on the provided client, datastore, and optional storage path.

    :param client: The MLClient object representing the Azure subscription.
    :type client: MLClient
    :param datastore: The Datastore object representing the Azure datastore.
    :type datastore: Datastore
    :param storage_path: The optional storage path within the datastore.
    :type storage_path: str, optional
    :return: The constructed Azure data path.
    :rtype: str
    """
    sub_f = f'subscriptions/{client.subscription_id}'
    rg_f = f'resourcegroups/{client.resource_group_name}'
    ws_f = f'workspaces/{client.workspace_name}'
    ds_f = f'datastores/{datastore.name}'
    if storage_path is None:
        return f'azureml://{sub_f}/{rg_f}/{ws_f}/{ds_f}'
    return f'azureml://{sub_f}/{rg_f}/{ws_f}/{ds_f}/paths/{storage_path}'

