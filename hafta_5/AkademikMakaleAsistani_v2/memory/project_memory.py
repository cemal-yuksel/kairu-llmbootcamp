"""
Project Memory - Advanced project-based memory management system
Manages multiple research projects, their resources, and cross-project insights
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchProject:
    """Data class for research project information"""
    project_id: str
    name: str
    description: str
    created_at: str
    last_updated: str
    status: str  # active, paused, completed, archived
    tags: List[str]
    documents: List[str]
    research_questions: List[str]
    key_findings: List[str]
    collaborators: List[str]
    deadline: Optional[str] = None
    progress_percentage: int = 0

@dataclass
class ProjectResource:
    """Data class for project resources (papers, notes, etc.)"""
    resource_id: str
    project_id: str
    name: str
    type: str  # pdf, note, link, data
    path: str
    added_at: str
    tags: List[str]
    summary: str
    importance: int  # 1-5 scale

class ProjectMemory:
    """
    Advanced project memory system that:
    - Manages multiple research projects simultaneously
    - Tracks project progress and milestones
    - Maintains project-specific resources and findings
    - Identifies cross-project connections and insights
    - Supports project collaboration and sharing
    """
    
    def __init__(self, memory_dir: str = None):
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).parent.parent / 'data' / 'projects'
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.projects_index_file = self.memory_dir / "projects_index.json"
        self.resources_index_file = self.memory_dir / "resources_index.json"
        self.connections_file = self.memory_dir / "project_connections.json"
        
        # In-memory storage
        self.projects: Dict[str, ResearchProject] = {}
        self.resources: Dict[str, ProjectResource] = {}
        self.project_connections: Dict[str, List[str]] = {}
        
        # Load existing data
        self._load_projects()
        self._load_resources()
        self._load_connections()
        
        logger.info("Project Memory system initialized")
    
    def _load_projects(self):
        """Load existing projects from storage"""
        try:
            if self.projects_index_file.exists():
                with open(self.projects_index_file, 'r', encoding='utf-8') as f:
                    projects_data = json.load(f)
                    
                for project_id, project_data in projects_data.items():
                    self.projects[project_id] = ResearchProject(**project_data)
                    
                logger.info(f"Loaded {len(self.projects)} projects")
        except Exception as e:
            logger.error(f"Error loading projects: {e}")
    
    def _save_projects(self):
        """Save projects to storage"""
        try:
            projects_data = {
                project_id: asdict(project) 
                for project_id, project in self.projects.items()
            }
            
            with open(self.projects_index_file, 'w', encoding='utf-8') as f:
                json.dump(projects_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving projects: {e}")
    
    def _load_resources(self):
        """Load existing resources from storage"""
        try:
            if self.resources_index_file.exists():
                with open(self.resources_index_file, 'r', encoding='utf-8') as f:
                    resources_data = json.load(f)
                    
                for resource_id, resource_data in resources_data.items():
                    self.resources[resource_id] = ProjectResource(**resource_data)
                    
                logger.info(f"Loaded {len(self.resources)} resources")
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
    
    def _save_resources(self):
        """Save resources to storage"""
        try:
            resources_data = {
                resource_id: asdict(resource)
                for resource_id, resource in self.resources.items()
            }
            
            with open(self.resources_index_file, 'w', encoding='utf-8') as f:
                json.dump(resources_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving resources: {e}")
    
    def _load_connections(self):
        """Load project connections from storage"""
        try:
            if self.connections_file.exists():
                with open(self.connections_file, 'r', encoding='utf-8') as f:
                    self.project_connections = json.load(f)
        except Exception as e:
            logger.error(f"Error loading connections: {e}")
    
    def _save_connections(self):
        """Save project connections to storage"""
        try:
            with open(self.connections_file, 'w', encoding='utf-8') as f:
                json.dump(self.project_connections, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving connections: {e}")
    
    def create_project(self, name: str, description: str, 
                      tags: List[str] = None, deadline: str = None) -> str:
        """
        Create a new research project
        
        Returns:
            project_id: Unique identifier for the created project
        """
        try:
            project_id = str(uuid.uuid4())
            
            project = ResearchProject(
                project_id=project_id,
                name=name,
                description=description,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                status="active",
                tags=tags or [],
                documents=[],
                research_questions=[],
                key_findings=[],
                collaborators=[],
                deadline=deadline,
                progress_percentage=0
            )
            
            self.projects[project_id] = project
            self._save_projects()
            
            # Create project directory
            project_dir = self.memory_dir / project_id
            project_dir.mkdir(exist_ok=True)
            
            logger.info(f"Created project: {name} (ID: {project_id})")
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return ""
    
    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get project by ID"""
        return self.projects.get(project_id)
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update project information"""
        try:
            if project_id not in self.projects:
                logger.warning(f"Project not found: {project_id}")
                return False
            
            project = self.projects[project_id]
            
            # Update allowed fields
            for key, value in updates.items():
                if hasattr(project, key):
                    setattr(project, key, value)
            
            project.last_updated = datetime.now().isoformat()
            
            self._save_projects()
            logger.info(f"Updated project: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating project: {e}")
            return False
    
    def add_resource(self, project_id: str, name: str, resource_type: str,
                    path: str, tags: List[str] = None, summary: str = "",
                    importance: int = 3) -> str:
        """Add a resource to a project"""
        try:
            if project_id not in self.projects:
                logger.warning(f"Project not found: {project_id}")
                return ""
            
            resource_id = str(uuid.uuid4())
            
            resource = ProjectResource(
                resource_id=resource_id,
                project_id=project_id,
                name=name,
                type=resource_type,
                path=path,
                added_at=datetime.now().isoformat(),
                tags=tags or [],
                summary=summary,
                importance=importance
            )
            
            self.resources[resource_id] = resource
            
            # Add to project's documents list
            project = self.projects[project_id]
            if resource_id not in project.documents:
                project.documents.append(resource_id)
                project.last_updated = datetime.now().isoformat()
            
            self._save_resources()
            self._save_projects()
            
            logger.info(f"Added resource to project {project_id}: {name}")
            return resource_id
            
        except Exception as e:
            logger.error(f"Error adding resource: {e}")
            return ""
    
    def get_project_resources(self, project_id: str) -> List[ProjectResource]:
        """Get all resources for a project"""
        try:
            project = self.projects.get(project_id)
            if not project:
                return []
            
            return [
                self.resources[resource_id] 
                for resource_id in project.documents 
                if resource_id in self.resources
            ]
            
        except Exception as e:
            logger.error(f"Error getting project resources: {e}")
            return []
    
    def add_finding(self, project_id: str, finding: str, source: str = "") -> bool:
        """Add a key finding to a project"""
        try:
            if project_id not in self.projects:
                logger.warning(f"Project not found: {project_id}")
                return False
            
            project = self.projects[project_id]
            
            finding_entry = {
                "finding": finding,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            
            project.key_findings.append(json.dumps(finding_entry))
            project.last_updated = datetime.now().isoformat()
            
            self._save_projects()
            self._update_cross_project_connections(project_id, finding)
            
            logger.info(f"Added finding to project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding finding: {e}")
            return False
    
    def _update_cross_project_connections(self, project_id: str, content: str):
        """Update connections between projects based on content similarity"""
        try:
            # Simple keyword-based connection detection
            content_words = set(content.lower().split())
            
            for other_project_id, other_project in self.projects.items():
                if other_project_id == project_id:
                    continue
                
                # Check for common keywords in findings and questions
                other_content = " ".join(other_project.key_findings + other_project.research_questions).lower()
                other_words = set(other_content.split())
                
                # Calculate similarity (simple Jaccard index)
                intersection = content_words.intersection(other_words)
                union = content_words.union(other_words)
                
                if len(union) > 0 and len(intersection) / len(union) > 0.1:  # 10% similarity threshold
                    # Add connection
                    if project_id not in self.project_connections:
                        self.project_connections[project_id] = []
                    
                    if other_project_id not in self.project_connections[project_id]:
                        self.project_connections[project_id].append(other_project_id)
                        
                    # Bidirectional connection
                    if other_project_id not in self.project_connections:
                        self.project_connections[other_project_id] = []
                    
                    if project_id not in self.project_connections[other_project_id]:
                        self.project_connections[other_project_id].append(project_id)
            
            self._save_connections()
            
        except Exception as e:
            logger.error(f"Error updating cross-project connections: {e}")
    
    def get_connected_projects(self, project_id: str) -> List[ResearchProject]:
        """Get projects connected to the given project"""
        try:
            connected_ids = self.project_connections.get(project_id, [])
            return [
                self.projects[pid] for pid in connected_ids 
                if pid in self.projects
            ]
        except Exception as e:
            logger.error(f"Error getting connected projects: {e}")
            return []
    
    def search_projects(self, query: str, tags: List[str] = None, 
                       status: str = None) -> List[ResearchProject]:
        """Search projects by query, tags, and status"""
        try:
            results = []
            query_lower = query.lower() if query else ""
            
            for project in self.projects.values():
                # Text search
                if query_lower and query_lower not in (
                    project.name.lower() + " " + project.description.lower()
                ):
                    continue
                
                # Tag filter
                if tags and not any(tag in project.tags for tag in tags):
                    continue
                
                # Status filter
                if status and project.status != status:
                    continue
                
                results.append(project)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching projects: {e}")
            return []
    
    def get_project_analytics(self, project_id: str) -> Dict[str, Any]:
        """Get analytics for a specific project"""
        try:
            project = self.projects.get(project_id)
            if not project:
                return {}
            
            resources = self.get_project_resources(project_id)
            connected_projects = self.get_connected_projects(project_id)
            
            # Calculate project duration
            created = datetime.fromisoformat(project.created_at)
            updated = datetime.fromisoformat(project.last_updated)
            duration_days = (updated - created).days
            
            analytics = {
                "project_info": {
                    "name": project.name,
                    "status": project.status,
                    "progress": project.progress_percentage,
                    "duration_days": duration_days
                },
                "resources": {
                    "total_resources": len(resources),
                    "resource_types": {},
                    "avg_importance": 0
                },
                "research_activity": {
                    "research_questions": len(project.research_questions),
                    "key_findings": len(project.key_findings),
                    "connected_projects": len(connected_projects)
                },
                "timeline": {
                    "created_at": project.created_at,
                    "last_updated": project.last_updated,
                    "deadline": project.deadline
                }
            }
            
            # Resource analytics
            if resources:
                resource_types = {}
                importance_sum = 0
                
                for resource in resources:
                    resource_types[resource.type] = resource_types.get(resource.type, 0) + 1
                    importance_sum += resource.importance
                
                analytics["resources"]["resource_types"] = resource_types
                analytics["resources"]["avg_importance"] = importance_sum / len(resources)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting project analytics: {e}")
            return {}
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get global analytics across all projects"""
        try:
            total_projects = len(self.projects)
            total_resources = len(self.resources)
            
            # Project status distribution
            status_dist = {}
            for project in self.projects.values():
                status_dist[project.status] = status_dist.get(project.status, 0) + 1
            
            # Resource type distribution
            resource_type_dist = {}
            for resource in self.resources.values():
                resource_type_dist[resource.type] = resource_type_dist.get(resource.type, 0) + 1
            
            # Most active projects (by recent updates)
            recent_projects = sorted(
                self.projects.values(),
                key=lambda p: p.last_updated,
                reverse=True
            )[:5]
            
            analytics = {
                "overview": {
                    "total_projects": total_projects,
                    "total_resources": total_resources,
                    "total_connections": sum(len(connections) for connections in self.project_connections.values()) // 2
                },
                "distributions": {
                    "project_status": status_dist,
                    "resource_types": resource_type_dist
                },
                "activity": {
                    "most_recent_projects": [
                        {"name": p.name, "last_updated": p.last_updated} 
                        for p in recent_projects
                    ]
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting global analytics: {e}")
            return {}
    
    def export_project(self, project_id: str) -> Dict[str, Any]:
        """Export complete project data"""
        try:
            project = self.projects.get(project_id)
            if not project:
                return {}
            
            resources = self.get_project_resources(project_id)
            connected_projects = self.get_connected_projects(project_id)
            
            export_data = {
                "project": asdict(project),
                "resources": [asdict(resource) for resource in resources],
                "connected_projects": [asdict(cp) for cp in connected_projects],
                "export_timestamp": datetime.now().isoformat()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting project: {e}")
            return {}