variable "aws_region" {
  type        = string
  description = "AWS region for the EC2 instance"
  default     = "us-east-1"
}

variable "instance_name" {
  type        = string
  description = "Name tag for the EC2 instance"
  default     = "carrb-video-ec2"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type"
  default     = "t3.micro"
}

variable "deep_learning_ami_ssm_parameter" {
  type        = string
  description = "Public SSM parameter name for the Deep Learning GPU AMI ID"
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}

variable "key_name" {
  type        = string
  description = "Optional EC2 key pair name. Leave null when using EC2 Instance Connect."
  default     = null
}

variable "ssh_cidr" {
  type        = string
  description = "CIDR allowed to SSH into the instance"
  default     = "0.0.0.0/0"
}

variable "root_volume_size_gb" {
  type        = number
  description = "Root EBS volume size in GB"
  default     = 16
}

variable "s3_bucket_arn" {
  type        = string
  description = "ARN of the S3 bucket this EC2 instance can access"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to resources"
  default     = {}
}
