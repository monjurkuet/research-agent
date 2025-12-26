# Variables
PYTHON = .venv/bin/python
MAIN = main.py

.PHONY: run save update clean

# 🚀 Launch the agent
run:
	@echo "--- Starting Bitcoin Research Agent ---"
	@ 

# 💾 Commit and Push changes to GitHub
save:
	@git add .
	@read -p "Commit message: " msg; 	git commit -m "$$msg"; 	git push origin main

# 🔄 Install/Update dependencies using UV
update:
	@uv sync

# 🧹 Clean up cache and memory
clean:
	rm -rf __pycache__ agent_memory.db
