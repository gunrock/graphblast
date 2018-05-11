#Makefile to fetch and install graph data for regression
#testing borrowed from Royal Caliber

#Each graph lives in its own directory
SUBDIRS = ASIC_320k c-big circuit5M citationCiteseer FullChip ins2 neos3 Stanford web-Stanford wheel_601

.PHONY: $(GRAPHS)

#fetches all graphs, extracts and sets up files for tests
all: recurse

#only download the graphs, but do not proceed further
fetch: recurse

#clean everything except the downloaded graphs
clean: recurse

#clean everything including the downloaded graphs
realclean: recurse

#recurse into each subdirectory and try to build the provided targets
recurse:
	for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir $(MAKECMDGOALS); done


$(GRAPHS):
	$(MAKE) -C $@
