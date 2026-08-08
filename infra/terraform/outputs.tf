output "api_url" {
  value       = "http://localhost:${var.api_port}"
  description = "FastAPI base URL"
}

output "ui_url" {
  value       = "http://localhost:${var.ui_port}"
  description = "Streamlit UI URL"
}

output "container_names" {
  value = [
    docker_container.api.name,
    docker_container.ui.name
  ]
  description = "Managed Docker container names"
}
