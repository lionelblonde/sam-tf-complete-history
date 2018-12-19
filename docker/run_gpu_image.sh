docker run -i -t --rm -v $(pwd)/MJKEY:/MJKEY \
                      -v $(pwd)/DEMOS:/DEMOS \
                      -v $(pwd)/data:/code/sam-tf/data docker-sam-tf-gpu:latest bash
