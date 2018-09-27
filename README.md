### Installation

```bash
cd minerva-scripts
pip install -r requirements.txt -e .
```

### Elastic Beanstalk Deployment

The application must be able to read the `metadata.xml` files for minerva.

In order to allow the above action, a policy at least as permisive as the following must be attached to `aws-elasticbeanstalk-ec2-role` or the role under which the elastic beanstalk instance is deployed:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:GetObjectVersionTorrent",
                "s3:GetObjectAcl",
                "s3:GetObject",
                "s3:GetObjectTorrent",
                "s3:GetObjectVersionTagging",
                "s3:GetObjectVersionAcl",
                "s3:GetObjectTagging",
                "s3:GetObjectVersionForReplication",
                "s3:GetObjectVersion",
                "s3:ListMultipartUploadParts"
            ],
            "Resource": "arn:aws:s3:::minerva-test-cf-common-tilebucket-*/*/metadata.xml"
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucketByTags",
                "s3:GetLifecycleConfiguration",
                "s3:ListBucketMultipartUploads",
                "s3:GetBucketTagging",
                "s3:GetInventoryConfiguration",
                "s3:GetBucketWebsite",
                "s3:ListBucketVersions",
                "s3:GetBucketLogging",
                "s3:GetAccelerateConfiguration",
                "s3:GetBucketVersioning",
                "s3:GetBucketAcl",
                "s3:GetBucketNotification",
                "s3:GetBucketPolicy",
                "s3:GetReplicationConfiguration",
                "s3:GetEncryptionConfiguration",
                "s3:GetBucketRequestPayment",
                "s3:GetBucketCORS",
                "s3:GetAnalyticsConfiguration",
                "s3:GetMetricsConfiguration",
                "s3:GetBucketLocation",
                "s3:GetIpConfiguration"
            ],
            "Resource": "arn:aws:s3:::minerva-test-cf-common-tilebucket-*"
        }
    ]
}
```

The application must be able to read and write `json` files storing figure layouts.

In order to allow the above action, a policy at least as permisive as the following must be attached to `aws-elasticbeanstalk-ec2-role` or the role under which the elastic beanstalk instance is deployed:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:PutAnalyticsConfiguration",
                "s3:GetObjectVersionTagging",
                "s3:CreateBucket",
                "s3:ReplicateObject",
                "s3:GetObjectAcl",
                "s3:DeleteBucketWebsite",
                "s3:PutLifecycleConfiguration",
                "s3:GetObjectVersionAcl",
                "s3:PutObjectTagging",
                "s3:DeleteObject",
                "s3:GetIpConfiguration",
                "s3:DeleteObjectTagging",
                "s3:GetBucketWebsite",
                "s3:PutReplicationConfiguration",
                "s3:DeleteObjectVersionTagging",
                "s3:GetBucketNotification",
                "s3:PutBucketCORS",
                "s3:GetReplicationConfiguration",
                "s3:ListMultipartUploadParts",
                "s3:PutObject",
                "s3:GetObject",
                "s3:PutBucketNotification",
                "s3:PutBucketLogging",
                "s3:GetAnalyticsConfiguration",
                "s3:GetObjectVersionForReplication",
                "s3:GetLifecycleConfiguration",
                "s3:ListBucketByTags",
                "s3:GetInventoryConfiguration",
                "s3:GetBucketTagging",
                "s3:PutAccelerateConfiguration",
                "s3:DeleteObjectVersion",
                "s3:GetBucketLogging",
                "s3:ListBucketVersions",
                "s3:ReplicateTags",
                "s3:RestoreObject",
                "s3:ListBucket",
                "s3:GetAccelerateConfiguration",
                "s3:GetBucketPolicy",
                "s3:PutEncryptionConfiguration",
                "s3:GetEncryptionConfiguration",
                "s3:GetObjectVersionTorrent",
                "s3:AbortMultipartUpload",
                "s3:PutBucketTagging",
                "s3:GetBucketRequestPayment",
                "s3:GetObjectTagging",
                "s3:GetMetricsConfiguration",
                "s3:DeleteBucket",
                "s3:PutBucketVersioning",
                "s3:ListBucketMultipartUploads",
                "s3:PutMetricsConfiguration",
                "s3:PutObjectVersionTagging",
                "s3:GetBucketVersioning",
                "s3:GetBucketAcl",
                "s3:PutInventoryConfiguration",
                "s3:PutIpConfiguration",
                "s3:GetObjectTorrent",
                "s3:PutBucketWebsite",
                "s3:PutBucketRequestPayment",
                "s3:GetBucketCORS",
                "s3:GetBucketLocation",
                "s3:ReplicateDelete",
                "s3:GetObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::minerva-figure-figures/*",
                "arn:aws:s3:::minerva-figure-figures"
            ]
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": [
                "s3:ListAllMyBuckets",
                "s3:HeadBucket"
            ],
            "Resource": "*"
        }
    ]
}
```

### Updating Omero

After logging in to `https://omero.hms.harvard.edu/figure/`, copy a cookie of this form:

```
_ga=GA*.*.*********.**********; _gid=GA*.*.*********.**********; csrftoken=********************************; sessionid=********************************
```

Assign the cookie to a variable and use it to download the files for OMERO.figure
```
OME_COOKIE=_ga=GA*.*.*********.**********; _gid=GA*.*.*********.**********; csrftoken=********************************; sessionid=********************************"

wget -r -p -U Mozilla https://omero.hms.harvard.edu/figure/ --header="Cookie: $OME_COOKIE" --no-check-certificate --content-disposition


rm -rf static
mv omero.hms.harvard.edu/static static
mv omero.hms.harvard.edu/figure/* static/figure/
for i in `find static -type f`; do mv $i `echo $i | cut -d? -f1`; done

cp -R open_with static/webgateway
```
