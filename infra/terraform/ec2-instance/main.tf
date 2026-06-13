terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_ssm_parameter" "deep_learning_gpu_ami" {
  name = var.deep_learning_ami_ssm_parameter
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

data "aws_iam_policy_document" "ec2_s3_access" {
  statement {
    sid    = "ListBucket"
    effect = "Allow"
    actions = [
      "s3:ListBucket"
    ]
    resources = [var.s3_bucket_arn]
  }

  statement {
    sid    = "ObjectReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = ["${var.s3_bucket_arn}/*"]
  }
}

resource "aws_iam_role" "ec2_s3_role" {
  name               = "${var.instance_name}-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = merge(
    {
      Name = "${var.instance_name}-role"
    },
    var.tags
  )
  permissions_boundary = "arn:aws:iam::783050088916:policy/UKDDCAWSRestrictedAdmin-PermBoundary"
}

resource "aws_iam_policy" "ec2_s3_policy" {
  name   = "${var.instance_name}-s3-policy"
  policy = data.aws_iam_policy_document.ec2_s3_access.json

  tags = merge(
    {
      Name = "${var.instance_name}-s3-policy"
    },
    var.tags
  )
}

resource "aws_iam_role_policy_attachment" "ec2_s3_attach" {
  role       = aws_iam_role.ec2_s3_role.name
  policy_arn = aws_iam_policy.ec2_s3_policy.arn
}

resource "aws_iam_instance_profile" "this" {
  name = "${var.instance_name}-instance-profile"
  role = aws_iam_role.ec2_s3_role.name

  tags = merge(
    {
      Name = "${var.instance_name}-instance-profile"
    },
    var.tags
  )
}

resource "aws_security_group" "ssh" {
  name        = "${var.instance_name}-ssh"
  description = "Allow SSH access (including EC2 Instance Connect)"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "${var.instance_name}-sg"
    },
    var.tags
  )
}

resource "aws_instance" "this" {
  ami                         = data.aws_ssm_parameter.deep_learning_gpu_ami.value
  instance_type               = var.instance_type
  key_name                    = var.key_name
  iam_instance_profile        = aws_iam_instance_profile.this.name
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  # Ensure the Instance Connect package is present on first boot.
  user_data = <<-EOT
    #!/bin/bash
    dnf install -y ec2-instance-connect || true
  EOT

  tags = merge(
    {
      Name = var.instance_name
    },
    var.tags
  )
}

output "instance_id" {
  value       = aws_instance.this.id
  description = "EC2 instance ID"
}

output "public_ip" {
  value       = aws_instance.this.public_ip
  description = "Public IP for the EC2 instance"
}

output "public_dns" {
  value       = aws_instance.this.public_dns
  description = "Public DNS for the EC2 instance"
}

output "security_group_id" {
  value       = aws_security_group.ssh.id
  description = "Security group ID used by EC2"
}

output "instance_role_name" {
  value       = aws_iam_role.ec2_s3_role.name
  description = "IAM role name attached to the EC2 instance"
}
