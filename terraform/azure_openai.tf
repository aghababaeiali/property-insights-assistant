resource "azurerm_cognitive_account" "openai" {
  name                = "property-insights-openai"
  resource_group_name = azurerm_resource_group.main.name
  location            = "swedencentral" # not every region has OpenAI models available
  kind                = "OpenAI"
  sku_name            = "S0"
}

resource "azurerm_cognitive_deployment" "gpt5_mini" {
  name                 = "gpt-5-mini"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = "gpt-5-mini"
    version = "2025-08-07" # check current deployable version before applying
  }
  # azurerm ~>3.100 (pinned in main.tf) calls this block "scale", not "sku" —
  # the sku rename came in the azurerm 4.x provider major version.
  scale {
    type     = "GlobalStandard"
    capacity = 10
  }
}

resource "azurerm_cognitive_deployment" "embeddings" {
  name                 = "text-embedding-3-small"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = "text-embedding-3-small"
    version = "1"
  }
  scale {
    type     = "GlobalStandard"
    capacity = 10
  }
}