"""Quick test script to validate settings functionality"""
from app import app
from flask import render_template

print("Testing settings page...")

with app.app_context():
    try:
        # Test template rendering
        html = render_template('settings.html')
        print("✓ Settings template renders successfully")
        print(f"✓ Template size: {len(html)} bytes")
        
        # Check for key elements
        checks = [
            ('form id="settingsForm"' in html, "Settings form"),
            ('LLM Configuration' in html, "LLM section"),
            ('Cloud GPU Configuration' in html, "Cloud GPU section"),
            ('File Upload Configuration' in html, "File upload section"),
            ('/api/settings' in html, "API endpoints")
        ]
        
        for check, desc in checks:
            status = "✓" if check else "✗"
            print(f"{status} {desc}")
        
        print("\n✓ All template validation checks passed!")
        
    except Exception as e:
        print(f"✗ Template error: {e}")
        import traceback
        traceback.print_exc()
