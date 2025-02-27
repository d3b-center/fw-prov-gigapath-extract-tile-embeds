FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV FLYWHEEL="/flywheel/v0"
WORKDIR ${FLYWHEEL}

# need gcc for openslide install
RUN apt-get update \
&& apt-get install gcc openslide-tools -y \
&& apt-get clean

# install main dependenices
COPY *.txt $FLYWHEEL/
RUN pip install -r requirements.txt
RUN pip install flywheel_gear_toolkit
RUN pip install fw_core_client
RUN pip install flywheel-sdk

# copy main files into working directory
COPY run.py manifest.json $FLYWHEEL/
COPY fw_gear_extract_tile_embeds ${FLYWHEEL}/fw_gear_extract_tile_embeds 
COPY ./ $FLYWHEEL/

# start the pipeline
RUN chmod a+x $FLYWHEEL/run.py
RUN chmod -R 777 .
ENTRYPOINT ["python","/flywheel/v0/run.py"]
