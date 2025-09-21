#!/usr/bin/env python3
"""
Test PDF generation to debug the issue.
"""

try:
    from fpdf import FPDF
    
    # Create a simple PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, 'Hello World!')
    
    # Test different output methods
    print("Testing PDF output methods...")
    
    # Method 1: Direct output
    pdf_data = pdf.output(dest='S')
    print(f"PDF output type: {type(pdf_data)}")
    print(f"PDF output length: {len(pdf_data)}")
    print(f"PDF starts with PDF: {pdf_data.startswith(b'%PDF')}")
    
    # Convert to bytes if needed
    if isinstance(pdf_data, bytearray):
        pdf_bytes = bytes(pdf_data)
    else:
        pdf_bytes = pdf_data
    print(f"Final bytes type: {type(pdf_bytes)}")
    print(f"Final bytes starts with PDF: {pdf_bytes.startswith(b'%PDF')}")
    
    # Save test PDF
    with open('test_output.pdf', 'wb') as f:
        f.write(pdf_bytes)
    
    print("✅ Test PDF saved as test_output.pdf")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()