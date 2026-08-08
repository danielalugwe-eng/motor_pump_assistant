locals {
  project_root = abspath("${path.module}/../..")
}

resource "docker_network" "mpps" {
  name = "mpps-network"
}

resource "docker_image" "api" {
  name = "mpps-api:latest"

  build {
    context    = local.project_root
    dockerfile = "Dockerfile.api"
  }
}

resource "docker_image" "ui" {
  name = "mpps-ui:latest"

  build {
    context    = local.project_root
    dockerfile = "Dockerfile.ui"
  }
}

resource "docker_container" "api" {
  name    = "mpps-api"
  image   = docker_image.api.image_id
  restart = "unless-stopped"

  networks_advanced {
    name = docker_network.mpps.name
  }

  env = [
    "OPENAI_API_KEY=${var.openai_api_key}"
  ]

  ports {
    internal = 8000
    external = var.api_port
  }

  volumes {
    host_path      = abspath("${local.project_root}/data")
    container_path = "/app/data"
  }

  volumes {
    host_path      = abspath("${local.project_root}/models")
    container_path = "/app/models"
  }
}

resource "docker_container" "ui" {
  name    = "mpps-ui"
  image   = docker_image.ui.image_id
  restart = "unless-stopped"

  networks_advanced {
    name = docker_network.mpps.name
  }

  env = [
    "API_BASE_URL=http://mpps-api:8000"
  ]

  ports {
    internal = 8501
    external = var.ui_port
  }

  volumes {
    host_path      = abspath("${local.project_root}/data")
    container_path = "/app/data"
  }

  volumes {
    host_path      = abspath("${local.project_root}/models")
    container_path = "/app/models"
  }

  depends_on = [docker_container.api]
}
