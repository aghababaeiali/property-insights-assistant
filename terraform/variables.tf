variable "resource_group_name" {
  description = "Name of the resource group containing all project resources"
  type        = string
  default     = "property-insights-rg"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "northeurope"
}

variable "postgres_admin_login" {
  description = "Admin username for the Postgres Flexible Server"
  type        = string
  default     = "pgadmin"
}

variable "postgres_admin_password" {
  description = "Admin password for the Postgres Flexible Server"
  type        = string
  sensitive   = true
}

variable "acr_name" {
  description = "Globally unique name for the Azure Container Registry"
  type        = string
  default     = "propertyinsightsacr"
}

variable "database_url" {
  description = "Full Postgres connection string for the app to use"
  type        = string
  sensitive   = true
}

variable "azure_openai_api_version" {
  description = "API version the openai SDK's AzureOpenAI client targets (not an Azure resource attribute, so it isn't derivable from azure_openai.tf — pinned here instead)"
  type        = string
  default     = "2024-10-21"
}

# Azure AI Search and Azure ML are deliberately outside this Terraform config
# (see ARCHITECTURE.md's Terraform section) — there's no local resource to
# read these from, so unlike AZURE_OPENAI_*, they're plain input variables,
# the same pattern already used for database_url.
variable "azure_search_endpoint" {
  description = "Azure AI Search endpoint URL (resource provisioned via az CLI, not this config)"
  type        = string
}

variable "azure_search_api_key" {
  description = "Azure AI Search admin/query API key"
  type        = string
  sensitive   = true
}

variable "azure_search_index" {
  description = "Azure AI Search index name used by agent/retriever.py"
  type        = string
}

variable "azure_ml_risk_endpoint_url" {
  description = "Azure ML managed online endpoint URL for the cancellation-risk model (resource provisioned via az CLI, not this config)"
  type        = string
}

variable "azure_ml_risk_endpoint_key" {
  description = "Azure ML managed online endpoint auth key"
  type        = string
  sensitive   = true
}