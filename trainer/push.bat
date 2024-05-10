docker build -t "tag-trainer" .
docker tag tag-trainer 339713009859.dkr.ecr.us-east-1.amazonaws.com/tag-trainer
aws ecr get-login-password | docker login --username AWS --password-stdin 339713009859.dkr.ecr.us-east-1.amazonaws.com
docker push 339713009859.dkr.ecr.us-east-1.amazonaws.com/tag-trainer