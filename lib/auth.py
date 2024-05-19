import os
from azureml.core.authentication import ServicePrincipalAuthentication
from azure.ai.ml import MLClient
from dotenv import load_dotenv






def __authenticator() -> ServicePrincipalAuthentication:
    """
    Return an instance of ServicePrincipalAuthentication.

    :return: An instance of ServicePrincipalAuthentication.
    """
    load_dotenv()
    return ServicePrincipalAuthentication(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        service_principal_id=os.environ["AZURE_CLIENT_ID"],
        service_principal_password=os.environ["AZURE_CLIENT_SECRET"])


def get_client() -> MLClient:
    """
    Returns an MLClient object configured with the Azure subscription ID, resource group, and workspace
    retrieved from the environment variables.

    :return: An MLClient object configured with the Azure subscription ID, resource group, and workspace
    :rtype: MLClient
    """
    load_dotenv()
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    workspace = os.environ["AZURE_WORKSPACE"]
    return MLClient(__authenticator(), subscription_id, resource_group, workspace)