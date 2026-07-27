resource "azurerm_log_analytics_workspace" "main" {
  name                = "workspace-property-insights"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "main" {
  name                       = "property-insights-env"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

resource "azurerm_user_assigned_identity" "acr_pull" {
  name                = "property-insights-acr-identity"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.acr_pull.principal_id
}

resource "azurerm_container_app" "main" {
  name                         = "property-insights-app"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.acr_pull.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.acr_pull.id
  }

  template {
    container {
      name   = "property-insights-agent"
      image  = "${azurerm_container_registry.main.login_server}/property-insights-agent:latest"
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "LLM_PROVIDER"
        value = "azure"
      }
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }

      # AZURE_OPENAI_* are read straight off the azurerm_cognitive_account /
      # azurerm_cognitive_deployment resources this same config creates
      # (azure_openai.tf) rather than duplicated into their own variables —
      # Terraform already owns those values, so a second, independently
      # maintained copy would just be one more place for the two to drift.
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = azurerm_cognitive_account.openai.endpoint
      }
      env {
        name        = "AZURE_OPENAI_API_KEY"
        secret_name = "azure-openai-api-key"
      }
      env {
        name  = "AZURE_OPENAI_DEPLOYMENT"
        value = azurerm_cognitive_deployment.gpt5_mini.name
      }
      env {
        name  = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        value = azurerm_cognitive_deployment.embeddings.name
      }
      env {
        name  = "AZURE_OPENAI_API_VERSION"
        value = var.azure_openai_api_version
      }

      # Azure AI Search and the Azure ML risk endpoint are provisioned
      # outside this config (see ARCHITECTURE.md), so — unlike the OpenAI
      # values above — there's no local resource to read these from; they
      # come in as plain variables, same pattern as database_url.
      env {
        name  = "AZURE_SEARCH_ENDPOINT"
        value = var.azure_search_endpoint
      }
      env {
        name        = "AZURE_SEARCH_API_KEY"
        secret_name = "azure-search-api-key"
      }
      env {
        name  = "AZURE_SEARCH_INDEX"
        value = var.azure_search_index
      }
      env {
        name  = "AZURE_ML_RISK_ENDPOINT_URL"
        value = var.azure_ml_risk_endpoint_url
      }
      env {
        name        = "AZURE_ML_RISK_ENDPOINT_KEY"
        secret_name = "azure-ml-risk-endpoint-key"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  secret {
    name  = "database-url"
    value = var.database_url
  }
  secret {
    name  = "azure-openai-api-key"
    value = azurerm_cognitive_account.openai.primary_access_key
  }
  secret {
    name  = "azure-search-api-key"
    value = var.azure_search_api_key
  }
  secret {
    name  = "azure-ml-risk-endpoint-key"
    value = var.azure_ml_risk_endpoint_key
  }
}