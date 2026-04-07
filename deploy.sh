#!/bin/sh

ssh -i VitalStaff.pem ec2-user@54.81.134.8 'cd VitalBites; git pull; cd frontend;REACT_APP_API_URL=https://www.vitalfoods.ai/api npm run build'
