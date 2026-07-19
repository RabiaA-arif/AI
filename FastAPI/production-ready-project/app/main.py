from contextlib import asynccontextmanager 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_setting
from app.database import sessionmanager

""" async await:in python async await enable the asynchoronous programing 
    allowing the tasks to run without blocking others 
    """

"""CORSMiddleware:
The CORSMiddleware in FastAPI is used to enable Cross-Origin
Resource Sharing (CORS), allowing your backend to handle requests
from different origins. This is essential when your
frontend and backend are hosted on separate domains or ports.
"""
    
