variable "aws_region" {
  type        = string
  description = "AWS region for the S3 bucket"
  default     = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name"
}

variable "force_destroy" {
  type        = bool
  description = "Allow bucket destroy even when it has objects"
  default     = false
}

variable "enable_versioning" {
  type        = bool
  description = "Enable S3 object versioning"
  default     = true
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to resources"
  default     = {}
}
