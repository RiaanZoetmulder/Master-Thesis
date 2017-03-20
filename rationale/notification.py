#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email Script
"""
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

def alert_user(user, password, message, title = 'Experiment Notification',
               attachment = None):
    
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = user
    msg['Subject'] = title
    msg.attach(MIMEText(message, 'plain'))
    
    
    
    if attachment:
        attach = open(attachment, "rb")
        
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attach).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" %\
                        attachment)
        msg.attach(part)
        
    
    text = msg.as_string()
    server = smtplib.SMTP_SSL('smtp.gmail.com')
    server.login(user, password)

    server.sendmail(user, user, text)
    
    server.quit()

