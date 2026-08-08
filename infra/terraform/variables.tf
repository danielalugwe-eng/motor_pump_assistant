variable "openai_api_key" {
  description = "OpenAI API key used by the FastAPI service"
  type        = string
  sensitive   = true
}

variable "api_port" {
  description = "Host port for FastAPI"
  type        = number
  default     = 8000
}

variable "ui_port" {
  description = "Host port for Streamlit UI"
  type        = number
  default     = 8501
}
