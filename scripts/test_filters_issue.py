"""
测试filters参数传递问题
"""

# 模拟langgraph_service.py中的filters处理逻辑
def test_filters_processing():
    """测试当前的filters处理逻辑"""
    
    print("\n" + "=" * 80)
    print("🔍 测试filters参数处理逻辑")
    print("=" * 80 + "\n")
    
    # 模拟用户传入的filters（和curl测试一致）
    input_filters = {
        "custom_filters": {
            "kb_ids": ["a86ea986-70f5-43b6-b3b9-fa58838973fa"]
        }
    }
    
    print("📥 输入的filters:")
    print(f"   {input_filters}")
    print(f"   kb_ids: {input_filters.get('custom_filters', {}).get('kb_ids')}")
    
    # 当前langgraph_service.py中的处理逻辑（第217-224行）
    print("\n" + "-" * 80)
    print("处理方式1: 当前langgraph_service.py的逻辑")
    print("-" * 80)
    
    processed_filters_v1 = (lambda f: (
        (lambda f2: (
            f2.update({"custom_filters": {}}) if not isinstance(f2.get("custom_filters"), dict) else None,
            f2
        )[-1])(f if isinstance(f, dict) else {})
    ))(input_filters)
    
    print(f"   处理后的filters: {processed_filters_v1}")
    print(f"   kb_ids: {processed_filters_v1.get('custom_filters', {}).get('kb_ids')}")
    
    # 正确的处理逻辑
    print("\n" + "-" * 80)
    print("处理方式2: 正确的处理逻辑")
    print("-" * 80)
    
    def process_filters_correctly(filters_input):
        """正确的filters处理"""
        # 确保filters是字典
        if not isinstance(filters_input, dict):
            filters_input = {}
        
        # 确保custom_filters存在且是字典
        if "custom_filters" not in filters_input:
            filters_input["custom_filters"] = {}
        elif not isinstance(filters_input.get("custom_filters"), dict):
            filters_input["custom_filters"] = {}
        
        return filters_input
    
    processed_filters_v2 = process_filters_correctly(input_filters.copy())
    print(f"   处理后的filters: {processed_filters_v2}")
    print(f"   kb_ids: {processed_filters_v2.get('custom_filters', {}).get('kb_ids')}")
    
    # 对比结果
    print("\n" + "=" * 80)
    print("📊 对比结果")
    print("=" * 80)
    
    v1_kb_ids = processed_filters_v1.get('custom_filters', {}).get('kb_ids')
    v2_kb_ids = processed_filters_v2.get('custom_filters', {}).get('kb_ids')
    
    print(f"\n处理方式1的kb_ids: {v1_kb_ids}")
    print(f"处理方式2的kb_ids: {v2_kb_ids}")
    
    if v1_kb_ids:
        print("\n✅ 处理方式1: kb_ids保留成功")
    else:
        print("\n❌ 处理方式1: kb_ids丢失！这就是问题所在！")
    
    if v2_kb_ids:
        print("✅ 处理方式2: kb_ids保留成功")
    else:
        print("❌ 处理方式2: kb_ids丢失")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    test_filters_processing()
