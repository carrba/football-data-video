# Terraform Deployments (Separated Lifecycle)

This folder contains two independent Terraform deployments:

- `s3-bucket/`: creates and manages a persistent S3 bucket.
- `ec2-instance/`: creates and manages a disposable EC2 instance.

Because they are separate directories, each has its own Terraform state and can be applied/destroyed independently.

## Prerequisites

- Terraform 1.6+
- AWS credentials configured (for example via `aws configure`, environment variables, or SSO)

## 1) Persistent S3 Bucket

```powershell
cd infra/terraform/s3-bucket
Copy-Item terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your globally unique bucket name
terraform init
terraform apply
```

Capture the bucket ARN output (you will use it in the EC2 deployment):

```powershell
terraform output bucket_arn
```

Notes:
- The bucket uses Terraform `prevent_destroy = true` to reduce accidental deletes.
- `force_destroy = false` prevents deleting a non-empty bucket.

## 2) Disposable EC2 Instance

```powershell
cd infra/terraform/ec2-instance
Copy-Item terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars (especially key_name, ssh_cidr, and s3_bucket_arn)
terraform init
terraform apply
```

The EC2 stack resolves the AMI from AWS Public SSM using a Deep Learning GPU AMI path by default.
You can override this in [infra/terraform/ec2-instance/terraform.tfvars](infra/terraform/ec2-instance/terraform.tfvars) with:

```hcl
deep_learning_ami_ssm_parameter = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
```

EC2 connectivity defaults to EC2 Instance Connect (no SSH key pair required):

- Keep `key_name = null` in `terraform.tfvars`.
- Keep port 22 open only to your IP (`ssh_cidr = "your.public.ip.address/32"`).
- Connect from AWS Console using Instance Connect, or from AWS CLI:

```powershell
aws ec2-instance-connect ssh --instance-id i-xxxxxxxx --os-user ec2-user
```

When you are done using the VM:

```powershell
cd infra/terraform/ec2-instance
terraform destroy
```

Destroying the EC2 deployment does not affect the S3 bucket deployment.

## Safety Recommendation

Set `ssh_cidr` to your current public IP in `/32` format rather than leaving `0.0.0.0/0`.
