#!/bin/sh

ssh -i VitalStaff.pem ec2-user@54.81.134.8 'set -e; cd VitalBites; git pull; ./venv/bin/pip install -r backend/requirements.txt; cd frontend; npm run build; cd ..; sudo systemctl restart vitalbites'
