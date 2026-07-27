output "container_app_url" {
  description = "Public URL of the deployed FastAPI app"
  value       = "https://${azurerm_container_app.main.latest_revision_fqdn}"
}

output "acr_login_server" {
  description = "Container registry hostname, for docker push/pull"
  value       = azurerm_container_registry.main.login_server
}