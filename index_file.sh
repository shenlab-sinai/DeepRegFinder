#!/bin/bash

bgzip $1
tabix -p bed $2