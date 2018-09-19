docker run -i -t --rm -v $(pwd)/MJKEY:/root/MJKEY \
                      -v $(pwd)/DEMOS:/root/DEMOS \
                      -v $(pwd)/data:/root/code/sam-tf/data docker-rl-tf-cpu:latest bash
