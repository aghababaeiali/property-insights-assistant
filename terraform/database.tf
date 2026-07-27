resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "property-insights-db"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  administrator_login    = var.postgres_admin_login
  administrator_password = var.postgres_admin_password

  sku_name   = "B_Standard_B1ms"
  storage_mb = 32768

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  # No high_availability block at all — that's how this provider version
  # represents "disabled" (its only valid `mode` values are ZoneRedundant/
  # SameZone; there's no explicit "Disabled" setting to name). Matches
  # today's actual choice: no standby replica, no zonal HA, deliberate for
  # a dev/portfolio workload, not an oversight.
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure_services" {
  name             = "AllowAllAzureServices"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_postgresql_flexible_server_database" "app_db" {
  name      = "property_insights"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}