.PHONY: clean data

# Make dataset
data:
	mkdir data ; mkdir data/raw; \
	ln -s /data/vision/polina/scratch/dmoyer/fseg/data/hcp_proc/ ./data/raw/ ; \
	cp data/raw/hcp_proc/1200_ids.csv data/raw/1200_ids.csv ; \

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
